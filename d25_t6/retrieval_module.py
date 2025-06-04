import copy
import math
import string
from typing import Any
import os
import ast
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from lightning import pytorch as pl
from transformers import RobertaTokenizer, RobertaModel

from d25_t6.beats import BEATsWrapper
from d25_t6.passt import CutInputIntoSegmentsWrapper


class AudioRetrievalModel(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters(kwargs)

        # audio encoder
        self.audio_embedding_model = CutInputIntoSegmentsWrapper(
            BEATsWrapper(
                checkpoint_path=kwargs['beats_ckpt_path']
            ),
            max_input_length=10*16000,
            segment_length=10*16000,
            hop_size=10*16000
        )
        self.audio_projection = torch.nn.Linear(768, 1024)

        # text encoder
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.text_embedding_model = RobertaModel.from_pretrained(
            'roberta-base' if kwargs['roberta_base'] else 'roberta-large',
            add_pooling_layer=False,
            hidden_dropout_prob=0.2,
            attention_probs_dropout_prob=0.2,
            output_hidden_states=False
        )
        self.text_projection = torch.nn.Linear(768 if kwargs['roberta_base'] else 1024, 1024)

        # Tau scheduling parameters (replaces learnable tau)
        self.tau_start = kwargs.get('tau_start', 0.2)
        self.tau_end = kwargs.get('tau_end', 0.05)
        self.current_tau = self.tau_start

        # Soft labeling parameters
        self.use_soft_labeling = kwargs.get('use_soft_labeling', True)
        self.similarity_threshold_start = kwargs.get('similarity_threshold_start', 0.8)
        self.similarity_threshold_end = kwargs.get('similarity_threshold_end', 0.65)
        self.soft_alpha = kwargs.get('soft_alpha', 2.0)
        self.soft_weight_warmup_epochs = kwargs.get('soft_weight_warmup_epochs', 3)
        self.soft_weight_transition_epochs = kwargs.get('soft_weight_transition_epochs', 8)
        self.soft_weight_final = kwargs.get('soft_weight_final', 0.7)

        self.validation_outputs = []
        self.kwargs = kwargs

        # Apply layer freezing if requested
        if kwargs.get('freeze_models', True):
            self.freeze_model_layers()

        self.compile_model()

    # Model freezing methods
    def freeze_beats_layers(self, num_layers_to_unfreeze=3):
        """Freeze all BEATs layers except the last N transformer layers"""
        beats_model = self.audio_embedding_model.model.model
        
        # Freeze all parameters first
        for param in beats_model.parameters():
            param.requires_grad = False
            
        # Unfreeze last N transformer layers in the encoder
        if hasattr(beats_model, 'encoder') and hasattr(beats_model.encoder, 'layers'):
            encoder_layers = beats_model.encoder.layers
            total_audio_layers = len(encoder_layers)
            
            # Unfreeze last N layers
            for layer in encoder_layers[-num_layers_to_unfreeze:]:
                for param in layer.parameters():
                    param.requires_grad = True
                    
            print(f"BEATs: Unfrozen last {num_layers_to_unfreeze} layers out of {total_audio_layers}")
        else:
            print("Warning: Could not find encoder.layers in BEATs model structure")
            
        # Keep audio projection layer trainable
        for param in self.audio_projection.parameters():
            param.requires_grad = True

    def freeze_roberta_layers(self, num_layers_to_unfreeze=3):
        """Freeze all RoBERTa layers except the last N layers"""
        # Freeze all parameters first
        for param in self.text_embedding_model.parameters():
            param.requires_grad = False
            
        # Unfreeze last N layers
        encoder_layers = self.text_embedding_model.encoder.layer
        total_layers = len(encoder_layers)
        
        for layer in encoder_layers[-num_layers_to_unfreeze:]:
            for param in layer.parameters():
                param.requires_grad = True
                
        print(f"RoBERTa: Unfrozen last {num_layers_to_unfreeze} layers out of {total_layers}")
        
        # Keep text projection layer trainable
        for param in self.text_projection.parameters():
            param.requires_grad = True

    def freeze_model_layers(self):
        """Apply layer freezing to both audio and text encoders"""
        beats_layers = self.kwargs.get('beats_unfreeze_layers', 3)
        roberta_layers = self.kwargs.get('roberta_unfreeze_layers', 3)
        
        print(f"Applying model freezing:")
        print(f"- BEATs: unfreezing last {beats_layers} layers")
        print(f"- RoBERTa: unfreezing last {roberta_layers} layers")
        
        self.freeze_beats_layers(beats_layers)
        self.freeze_roberta_layers(roberta_layers)

    # Tau scheduling methods
    def get_adaptive_tau(self):
        """Curriculum learning: start easy (high tau), become harder (low tau)"""
        total_epochs = self.trainer.max_epochs
        current_epoch = self.current_epoch
        
        # Cosine annealing
        progress = current_epoch / total_epochs
        tau = self.tau_end + (self.tau_start - self.tau_end) * 0.5 * (1 + math.cos(math.pi * progress))
        
        return tau

    def update_tau(self):
        """Update tau based on current training progress"""
        self.current_tau = self.get_adaptive_tau()

    # Soft labeling scheduling methods
    def get_soft_weight_schedule(self):
        """Gradually increase soft positive contribution"""
        current_epoch = self.current_epoch
        
        if current_epoch < self.soft_weight_warmup_epochs:
            return 0.0  # Start with only hard positives
        elif current_epoch < self.soft_weight_transition_epochs:
            # Linear interpolation during transition
            progress = (current_epoch - self.soft_weight_warmup_epochs) / (self.soft_weight_transition_epochs - self.soft_weight_warmup_epochs)
            return progress * self.soft_weight_final
        else:
            return self.soft_weight_final  # Full soft positive weight

    def get_similarity_threshold(self):
        """Start conservative, then relax threshold"""
        current_epoch = self.current_epoch
        total_epochs = self.trainer.max_epochs
        
        # Linear annealing from start to end threshold
        progress = min(current_epoch / total_epochs, 1.0)
        threshold = self.similarity_threshold_start - (self.similarity_threshold_start - self.similarity_threshold_end) * progress
        
        return max(self.similarity_threshold_end, threshold)

    # Soft labeling methods
    def compute_caption_similarities(self, batch):
        """Compute soft similarities between captions in the batch"""
        
        # Get captions
        captions = []
        for i, b in enumerate([c[0] for c in batch['captions']]):
            if not (type(b) == str):
                b = b[0]
            captions.append(b.lower().translate(str.maketrans('', '', string.punctuation)))

        # Tokenize captions
        tokenized = self.tokenizer(
            captions,
            add_special_tokens=True,
            padding='max_length',
            return_tensors='pt',
            max_length=32,
            truncation=True
        )

        # Get text embeddings before projection (for similarity computation)
        device = next(self.text_embedding_model.parameters()).device
        token_embeddings = self.text_embedding_model(
            input_ids=tokenized['input_ids'].to(device),
            attention_mask=tokenized['attention_mask'].to(device)
        )[0]
        
        # Use CLS token embeddings for similarity
        caption_embeddings = token_embeddings[:, 0, :]  # (batch_size, hidden_dim)
        caption_embeddings = F.normalize(caption_embeddings, p=2, dim=-1)
        
        # Compute pairwise similarities
        similarity_matrix = torch.matmul(caption_embeddings, caption_embeddings.T)
        
        return similarity_matrix

    def create_soft_positive_matrix(self, batch):
        """Create soft positive matrix combining filename matching and caption similarity"""
        
        # Original binary matrix (same audio file)
        paths = np.array([hash(batch['dataset'][i] + batch['subset'][i] + p) 
                         for i, p in enumerate(batch['fname'])])
        binary_matrix = torch.tensor(paths[None, :] == paths[:, None]).float().to(self.device)
        
        if not self.use_soft_labeling:
            return binary_matrix
        
        # Get current scheduling parameters
        soft_weight = self.get_soft_weight_schedule()
        similarity_threshold = self.get_similarity_threshold()
        
        # If no soft labeling yet, return binary matrix
        if soft_weight == 0.0:
            return binary_matrix
        
        # Caption similarity matrix
        caption_similarities = self.compute_caption_similarities(batch)
        
        # Create soft positive matrix
        soft_matrix = binary_matrix.clone()
        
        # Add soft positives (similar captions, different audio)
        soft_mask = (caption_similarities > similarity_threshold) & (binary_matrix == 0)
        
        if soft_mask.sum() > 0:
            # Apply sharpening and weighting
            soft_similarities = torch.pow(caption_similarities[soft_mask], self.soft_alpha)
            soft_matrix[soft_mask] = soft_weight * soft_similarities
        
        return soft_matrix

    def compute_soft_contrastive_loss(self, C, soft_positive_matrix):
        """Compute soft contrastive loss using weighted positive samples"""
        
        # Apply tau scaling
        C_scaled = C / self.current_tau
        
        # Compute log softmax
        C_audio = torch.log_softmax(C_scaled, dim=0)  # P(a|t)
        C_text = torch.log_softmax(C_scaled, dim=1)   # P(t|a)
        
        # Find positive pairs
        positive_mask = soft_positive_matrix > 0
        
        if positive_mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=C.device)
        
        # Weighted positive loss
        weights = soft_positive_matrix[positive_mask]
        
        # Audio-to-text loss (weighted by soft positives)
        audio_positive_loss = (C_audio[positive_mask] * weights).sum() / weights.sum()
        
        # Text-to-audio loss (weighted by soft positives)
        text_positive_loss = (C_text[positive_mask] * weights).sum() / weights.sum()
        
        loss = -0.5 * (audio_positive_loss + text_positive_loss)
        
        return loss

    def compile_model(self):
        """Apply torch.compile() if GPU is recent"""
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device)
            if properties.major >= 7 and self.kwargs.get('compile', False):
                print("Compiling Models")
                self.text_embedding_model = torch.compile(self.text_embedding_model)
                self.audio_embedding_model.model.model = torch.compile(self.audio_embedding_model.model.model)

    def forward(self, batch) -> Any:
        text_embeddings = self.forward_text(batch)
        audio_embeddings = self.forward_audio(batch)
        return audio_embeddings, text_embeddings

    def forward_audio(self, batch):
        print(f"Input audio shape: {batch['audio'].shape}")
        print(f"Input durations: {batch['duration']}")
        
        audio_input = batch['audio'].mean(1)
        print(f"Audio input to model shape: {audio_input.shape}")
        
        audio_embeddings = self.audio_embedding_model(audio_input)
        print(f"Audio embeddings shape from model: {audio_embeddings.shape}")

        # mask embeddings from padded empty audio parts
        aggregated = []
        for i, duration in enumerate(batch['duration']):
            if duration <= 10:
                emb = audio_embeddings[i, 0]
                print(f"Duration {duration:.2f}s, using first segment, shape: {emb.shape}")
                aggregated.append(emb)
            elif duration <= 20:
                emb = audio_embeddings[i, :2].mean(-2)
                print(f"Duration {duration:.2f}s, using first 2 segments, shape: {emb.shape}")
                aggregated.append(emb)
            else:
                emb = audio_embeddings[i].mean(-2)
                print(f"Duration {duration:.2f}s, using all segments, shape: {emb.shape}")
                aggregated.append(emb)

        audio_embeddings = torch.stack(aggregated)
        print(f"Stacked embeddings shape: {audio_embeddings.shape}")
        
        audio_embeddings = self.audio_projection(audio_embeddings)
        audio_embeddings = torch.nn.functional.normalize(audio_embeddings, p=2, dim=-1)
        return audio_embeddings

    def forward_text(self, batch):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        captions = []
        for i, b in enumerate([c[0] for c in batch['captions']]):
            if not (type(b) == str):
                print(b)
                b = b[0]
            captions.append(b.lower().translate(str.maketrans('', '', string.punctuation)))

        tokenized = self.tokenizer(
            captions,
            add_special_tokens=True,
            padding='max_length',
            return_tensors='pt',
            max_length=32,
            truncation=True
        )

        token_embeddings = self.text_embedding_model(
            input_ids=tokenized['input_ids'].to(device),
            attention_mask=tokenized['attention_mask'].to(device)
        )[0]
        
        sentence_features = token_embeddings[:, 0, :]
        sentence_features = self.text_projection(sentence_features)
        sentence_features = torch.nn.functional.normalize(sentence_features, p=2, dim=-1)

        return sentence_features

    def training_step(self, batch, batch_idx):
        self.lr_scheduler_step(batch_idx)
        
        # Update tau for current epoch
        self.update_tau()

        audio_embeddings, text_embeddings = self.forward(batch)

        # Compute pairwise similarities
        C = torch.matmul(audio_embeddings, text_embeddings.T)

        if self.use_soft_labeling:
            # Use soft positive matrix and soft contrastive loss
            soft_positive_matrix = self.create_soft_positive_matrix(batch)
            loss = self.compute_soft_contrastive_loss(C, soft_positive_matrix)
            
            # Log soft labeling metrics
            self.log('train/soft_weight', self.get_soft_weight_schedule(), sync_dist=True)
            self.log('train/similarity_threshold', self.get_similarity_threshold(), sync_dist=True)
            self.log('train/hard_positives', (soft_positive_matrix == 1.0).sum().float(), sync_dist=True)
            self.log('train/soft_positives', ((soft_positive_matrix > 0) & (soft_positive_matrix < 1.0)).sum().float(), sync_dist=True)
            
        else:
            # Original binary approach
            C_scaled = C / self.current_tau
            C_audio = torch.log_softmax(C_scaled, dim=0)
            C_text = torch.log_softmax(C_scaled, dim=1)
            
            paths = np.array([hash(batch['dataset'][i] + batch['subset'][i] + p) for i, p in enumerate(batch['fname'])])
            I = torch.tensor(paths[None, :] == paths[:, None]).to(self.device)
            
            loss = -0.5 * (C_audio[torch.where(I)].mean() + C_text[torch.where(I)].mean())

        self.log("train/loss", loss, batch_size=len(audio_embeddings), sync_dist=True, prog_bar=True)
        self.log('train/tau', self.current_tau, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        audio_embeddings, text_embeddings = self.forward(batch)

        args = {
            'audio_embeddings': copy.deepcopy(audio_embeddings.detach()),
            'text_embeddings': copy.deepcopy(text_embeddings.detach()),
            'caption': [c[0] for c in batch['captions']],
            'path': batch['fname']
        }

        self.validation_outputs.append(args)

    def on_validation_epoch_end(self, prefix='val'):
        outputs = self.validation_outputs

        # concatenate metadata
        paths = np.array([p for b in outputs for p in b['path']])
        captions = np.array([p for b in outputs for p in b['caption']])

        # audios in clotho can have five captions
        target = []
        select = []
        first_occurrence = {}
        for i, p in enumerate(paths):
            index = first_occurrence.get(p)
            if index is None:
                index = len(first_occurrence)
                first_occurrence[p] = index
                select.append(i)
            target.append(index)
        paths = paths[select]

        # concatenate embeddings
        audio_embeddings = torch.cat([o['audio_embeddings'] for o in outputs])[select]
        text_embeddings = torch.cat([o['text_embeddings'] for o in outputs])

        # concatenate global ranking
        C = torch.matmul(text_embeddings, audio_embeddings.T)

        # get top 10
        top_ten = C.topk(10, dim=1)[1].detach().cpu().numpy()
        target = np.array(target)

        # recall metrics
        r_1 = (top_ten[:, :1] == target[:, None]).sum(axis=1).mean()
        r_5 = (top_ten[:, :5] == target[:, None]).sum(axis=1).mean()
        r_10 = (top_ten == target[:, None]).sum(axis=1).mean()

        # mAP@10
        AP = 1 / ((top_ten == target[:, None]).argmax(axis=1) + 1)
        AP[~(top_ten == target[:, None]).any(axis=1)] = 0
        mAP = AP.mean()

        # log retrieval performance
        self.log(f'{prefix}/R@1', r_1)
        self.log(f'{prefix}/R@5', r_5)
        self.log(f'{prefix}/R@10', r_10)
        self.log(f'{prefix}/mAP@10', mAP)

        if os.path.exists(f'resources/metadata_eval.csv') and prefix == 'test':
            matched_files = pd.read_csv(f'resources/metadata_eval.csv')
            matched_files["audio_filenames"] = matched_files["audio_filenames"].transform(lambda x: ast.literal_eval(x))

            def get_ranks(c, r):
                ranks = [i.item() for i in torch.argsort(torch.argsort(-c))[r]]
                return ranks

            matched_files["query_index"] = matched_files["query"].transform(lambda x: captions.tolist().index(x))
            matched_files["new_audio_indices"] = matched_files["audio_filenames"].transform(lambda x: [paths.tolist().index(y) for y in x])
            matched_files["TP_ranks"] = matched_files.apply(lambda row: get_ranks(C[row["query_index"]], row["new_audio_indices"]), axis=1)

            def average_precision_at_k(relevant_ranks, k=10):
                relevant_ranks = sorted(relevant_ranks)
                ap = 0.0
                for i, rank in enumerate(relevant_ranks, start=1):
                    if rank >= k:
                        break
                    ap += i / (rank + 1)
                return ap / len(relevant_ranks)

            new_mAP = matched_files["TP_ranks"].apply(lambda ranks: average_precision_at_k(ranks, 10)).mean()
            self.log(f'{prefix}_multiple_positives/mAP@10', new_mAP)

        self.validation_outputs.clear()

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        self.on_validation_epoch_end(prefix='test')

    def configure_optimizers(self):
        # Only optimize parameters that require gradients
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        
        print(f"Training {len(trainable_params)} parameter groups out of {len(list(self.parameters()))} total")
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-5,
            amsgrad=False
        )

        return optimizer

    def lr_scheduler_step(self, batch_idx):
        steps_per_epoch = self.trainer.num_training_batches

        min_lr = self.kwargs['min_lr']
        max_lr = self.kwargs['max_lr']
        current_step = self.current_epoch * steps_per_epoch + batch_idx
        warmup_steps = self.kwargs['warmup_epochs'] * steps_per_epoch
        total_steps = (self.kwargs['warmup_epochs'] + self.kwargs['rampdown_epochs']) * steps_per_epoch
        decay_steps = total_steps - warmup_steps

        if current_step < warmup_steps:
            lr = min_lr + (max_lr - min_lr) * (current_step / warmup_steps)
        elif current_step < total_steps:
            decay_progress = (current_step - warmup_steps) / decay_steps
            lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * decay_progress))
        else:
            lr = min_lr

        for param_group in self.optimizers(use_pl_optimizer=False).param_groups:
            param_group['lr'] = lr

        self.log('train/lr', lr, sync_dist=True)
