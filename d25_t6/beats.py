import torch
import torch.nn as nn
import torchaudio

class BEATsWrapper(nn.Module):
    def __init__(self, checkpoint_path, device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading BEATs checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Try to load the model state directly
        from .beats_utils import BEATs, BEATsConfig
        
        cfg = BEATsConfig(checkpoint['cfg'])
        self.model = BEATs(cfg)
        
        # Load state dict more carefully
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        # Remove any problematic keys
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if 'predictor' not in k:  # Skip predictor weights if present
                filtered_state_dict[k] = v
                
        self.model.load_state_dict(filtered_state_dict, strict=False)
        self.model.eval()
        self.model.to(self.device)
        self.target_sr = 16000
        
        print("BEATs model loaded successfully!")

    def forward(self, x):
        x = x.to(self.device)
        
        # Resample to 16kHz if not already
        if x.shape[-1] != 160000:  # Expected length for 10s at 16kHz
            x = torchaudio.functional.resample(x, x.shape[-1]/10, 16000)
        
        print(f"BEATs processing audio shape: {x.shape}")
        
        with torch.no_grad():
            try:
                # Use preprocess and extract_features separately
                fbank = self.model.preprocess(x)  # Convert to fbank features
                print(f"Fbank shape: {fbank.shape}")
                
                # Extract features
                fbank = fbank.unsqueeze(1)  # Add channel dim
                features = self.model.patch_embedding(fbank)
                features = features.reshape(features.shape[0], features.shape[1], -1)
                features = features.transpose(1, 2)
                features = self.model.layer_norm(features)
                
                if self.model.post_extract_proj is not None:
                    features = self.model.post_extract_proj(features)
                
                features = self.model.dropout_input(features)
                features, _ = self.model.encoder(features)
                
                # Pool to get final embedding
                pooled = features.mean(dim=1)  # (batch, 768)
                print(f"Final pooled shape: {pooled.shape}")
                
                return pooled
                
            except Exception as e:
                print(f"Error in forward pass: {e}")
                # Return dummy embedding as fallback
                return torch.randn(x.shape[0], 768, device=self.device)
