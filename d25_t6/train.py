import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")
warnings.filterwarnings("ignore", category=FutureWarning, module="hear21passt.models.preprocess")

import os
from typing import Union, List, Mapping
import torch
import wandb
import argparse
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import seed_everything

from aac_datasets import Clotho, WavCaps, AudioCaps
from torch.utils.data import DataLoader
from d25_t6.datasets.download_datasets import download_clotho, download_audiocaps, download_wavcaps_mp3
from d25_t6.datasets.audio_loading import custom_loading
from d25_t6.datasets.utils import exclude_broken_files, exclude_forbidden_files, exclude_forbidden_and_long_files
from d25_t6.datasets.batch_collate import CustomCollate

from d25_t6.retrieval_module import AudioRetrievalModel

torch.set_float32_matmul_precision('high')  # or 'medium'

def train(
        model: AudioRetrievalModel,
        train_ds: torch.utils.data.Dataset,
        val_ds: torch.utils.data.Dataset,
        logger: Union[None, WandbLogger],
        args: dict
):
    """
    Trains the AudioRetrievalModel using provided datasets, logger, and configuration arguments.

    Args:
        model (d25_t6.retrieval_module.AudioRetrievalModel): The model to be trained.
        train_ds (torch.utils.data.Dataset): The training dataset.
        val_ds (torch.utils.data.Dataset): The validation dataset.
        logger (Union[None, WandbLogger]): The logger for tracking training metrics.
        args (dict): A dictionary of configuration arguments for training.

    Returns:
        d25_t6.retrieval_module.AudioRetrievalModel: The trained model.
    """
    # get a unique experiment name for name of checkpoint
    if wandb.run is not None:
        experiment_name = wandb.run.name or wandb.run.id  # Use name if available, else use ID
    else:
        experiment_name = "experiment_" + wandb.util.generate_id()  # Random unique ID fallback

    # create path for the model checkpoints
    checkpoint_dir = os.path.join(args["checkpoints_path"], experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure directory exists

    # checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="{epoch}",
        save_top_k=1,
        monitor="val/mAP@10",
        mode="max",
        save_last=True
    )

    # trainer
    trainer = pl.Trainer(
        devices=args['devices'],
        logger=logger if wandb.run else None,
        callbacks=[checkpoint_callback],
        max_epochs=args['max_epochs'],
        precision=args['precision'],
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        num_sanity_val_steps=0,
        fast_dev_run=False
    )

    ### train on training set; monitor performance on val
    trainer.fit(
        model,
        train_dataloaders=DataLoader(
            train_ds, batch_size=args['batch_size'], num_workers=args['n_workers'], shuffle=True, drop_last=True,
            persistent_workers=True, collate_fn=CustomCollate()
        ),
        val_dataloaders=DataLoader(
            val_ds, batch_size=args['batch_size_eval'], num_workers=args['n_workers'], shuffle=False, drop_last=False,
            persistent_workers=True, collate_fn=CustomCollate()
        ),
        ckpt_path=args['resume_ckpt_path'] # should be none unless training is resumed
    )

    return model

def test(
        model: AudioRetrievalModel,
        test_ds: torch.utils.data.Dataset,
        logger: Union[None, WandbLogger],
        args: dict
) -> List[Mapping[str, float]]:
    """
    Tests the trained AudioRetrievalModel on a given test dataset.

    Args:
        model (d25_t6.retrieval_module.AudioRetrievalModel): The trained model to be evaluated.
        test_ds (torch.utils.data.Dataset): The test dataset.
        logger (Union[None, WandbLogger]): The logger for tracking test metrics.
        args (dict): A dictionary of configuration arguments for testing.

    Returns:
        dict: The result of the model evaluation on the test dataset.
    """
    trainer = pl.Trainer(
        devices=args['devices'],
        logger=logger if wandb.run else None,
        callbacks=None,
        max_epochs=args['max_epochs'],
        precision=args['precision'],
        num_sanity_val_steps=0,
        fast_dev_run=False
    )

    ### test on the eval set
    result = trainer.test(
        model,
        DataLoader(
            test_ds, batch_size=args['batch_size_eval'], num_workers=args['n_workers'], shuffle=False, drop_last=False,
            persistent_workers=True, collate_fn=CustomCollate()
        )
    )

    return result


def get_args() -> dict:
    """
    Parses command-line arguments for configuring the training and testing process.

    Returns:
        dict: A dictionary containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Argument parser for training configuration.")

    parser.add_argument('--devices', type=str, default='auto', help='Device selection (e.g., auto, cpu, cuda, etc.)')
    parser.add_argument('--n_workers', type=int, default=16, help='Number of workers for data loading')
    parser.add_argument('--compile', default=False, action=argparse.BooleanOptionalAction, help='Compile the model if GPU version >= 7.')
    parser.add_argument('--logging', default=True, action=argparse.BooleanOptionalAction, help='Log metrics in wandb or not.')

    # Parameter initialization & resume training
    parser.add_argument('--resume_ckpt_path', type=str, default=None, help='Path to checkpoint to resume training from.')
    parser.add_argument('--load_ckpt_path', type=str, default=None, help='Path to checkpoint used as a weight initialization for training.')

    # Training parameters
    parser.add_argument('--seed', type=int, default=21208, help='Random seed of experiment')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--batch_size_eval', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--max_epochs', type=int, default=20, help='Maximum number of epochs')
    parser.add_argument('--warmup_epochs', type=int, default=1, help='Number of warmup epochs')
    parser.add_argument('--rampdown_epochs', type=int, default=15, help='Number of ramp-down epochs')
    parser.add_argument('--max_lr', type=float, default=1e-5, help='Maximum learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-7, help='Minimum learning rate')

    # Tau scheduling parameters (replaces old tau parameters)
    parser.add_argument('--tau_start', type=float, default=0.2, help='Starting tau value (easy learning)')
    parser.add_argument('--tau_end', type=float, default=0.05, help='Ending tau value (hard learning)')

    # Model freezing parameters
    parser.add_argument('--beats_unfreeze_layers', type=int, default=3, 
                       help='Number of last transformer layers to unfreeze in BEATs audio encoder (default: 3)')
    parser.add_argument('--roberta_unfreeze_layers', type=int, default=3, 
                       help='Number of last transformer layers to unfreeze in RoBERTa text encoder (default: 3)')
    parser.add_argument('--freeze_models', default=True, action=argparse.BooleanOptionalAction,
                       help='Whether to freeze pretrained models except last N layers (default: True)')

    # Soft labeling parameters
    parser.add_argument('--use_soft_labeling', default=True, action=argparse.BooleanOptionalAction,
                       help='Enable soft positive labeling based on caption similarities (default: True)')
    parser.add_argument('--similarity_threshold_start', type=float, default=0.85,
                       help='Initial similarity threshold for soft positives (default: 0.85)')
    parser.add_argument('--similarity_threshold_end', type=float, default=0.65,
                       help='Final similarity threshold for soft positives (default: 0.65)')
    parser.add_argument('--soft_alpha', type=float, default=2.0,
                       help='Exponent for sharpening similarity scores (default: 2.0)')
    parser.add_argument('--soft_weight_warmup_epochs', type=int, default=5,
                       help='Epochs before introducing soft positives (default: 5)')
    parser.add_argument('--soft_weight_transition_epochs', type=int, default=15,
                       help='Epochs to reach full soft positive weight (default: 15)')
    parser.add_argument('--soft_weight_final', type=float, default=0.3,
                       help='Final weight for soft positives (default: 0.3)')

    # RoBERTa parameters
    parser.add_argument('--roberta_base', default=False, action=argparse.BooleanOptionalAction,  help='Use Roberta base or large.')

    # use additional data sets
    parser.add_argument('--wavcaps', default=False, action=argparse.BooleanOptionalAction, help='Include WavCaps in the training or not.')
    parser.add_argument('--audiocaps', default=False, action=argparse.BooleanOptionalAction, help='Include AudioCaps in the training or not.')
    parser.add_argument('--ablate_clean_setup', default=True, action=argparse.BooleanOptionalAction, help='Include ClothoV2.1 eval, test in the training or not.')

    # Paths
    parser.add_argument('--data_path', type=str, default='data', help='Path to dataset; dataset will be downloaded into this folder.')
    parser.add_argument('--checkpoints_path', type=str, default='checkpoints', help='Path to save checkpoints to.')

    # run training / test
    parser.add_argument('--train', default=True, action=argparse.BooleanOptionalAction, help='Run training or not.')
    parser.add_argument('--test', default=True, action=argparse.BooleanOptionalAction, help='Run testing or not.')

    # BEATs parameters
    parser.add_argument('--beats_ckpt_path', type=str, required=True, help='Path to BEATs fine-tuned checkpoint (AS2M, cpt2).')

    parser.add_argument('--precision', type=str, default='16-mixed', help='Training precision (32, 16-mixed, bf16-mixed)')

    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    """
    Entry point for training and testing the model.
    - Downloads datasets if necessary.
    - Initializes logging and model.
    - Runs training and/or testing based on arguments.
    """
    args = get_args()

    os.makedirs(args["data_path"], exist_ok=True)
    # download data sets; will be ignored if exists
    # ClothoV2.1
    download_clotho(args["data_path"])
    # AudioCAps
    if args['audiocaps']:
        download_audiocaps(args["data_path"])
    # WavCaps
    if args['wavcaps']:
        download_wavcaps_mp3(args["data_path"])
        # download_wavcaps(args["data_path"], args["huggingface_cache_path"])

    # set a seed to make experiments reproducible
    if args['seed'] > 0:
        seed_everything(args['seed'], workers=True)
    else:
        print("Not seeding experiment.")

    # initialize wandb, i.e., the logging framework
    if args['logging']:
        wandb.init(project="d25_t6")
        logger = WandbLogger()
    else:
        logger = None

    # initialize the model
    if args['load_ckpt_path']:
        model = AudioRetrievalModel.load_from_checkpoint(args['load_ckpt_path'])
    else:
        model = AudioRetrievalModel(**args)

    # train
    if args['train']:
        # get training ad validation data sets; add the resampling transformation
        train_ds = custom_loading(Clotho(subset="dev", root=args["data_path"], flat_captions=True))

        if args['audiocaps']:
            ac = custom_loading(
                AudioCaps(subset="train", root=args["data_path"], download=True, download_audio=False, audio_format='mp3')
            )
            train_ds = torch.utils.data.ConcatDataset([train_ds, ac])

        if args['wavcaps']:
            # load the subsets
            wc_f = exclude_forbidden_files(custom_loading(WavCaps(subset="freesound", root=args["data_path"])))
            wc_b = custom_loading(WavCaps(subset="bbc", root=args["data_path"]))
            wc_s = custom_loading(WavCaps(subset="soundbible", root=args["data_path"]))
            wc_a = exclude_broken_files(custom_loading(WavCaps(subset="audioset_no_audiocaps" if not args["ablate_clean_setup"] else "audioset", root=args["data_path"])))
            train_ds = torch.utils.data.ConcatDataset([train_ds, wc_f, wc_b, wc_s, wc_a])

        val_ds = custom_loading(Clotho(subset="val", root=args["data_path"], flat_captions=True))

        model = train(model, train_ds, val_ds, logger, args)

    # test
    if args['test']:
        test_ds = custom_loading(Clotho(subset="eval", root=args["data_path"], flat_captions=True))

        results = test(model, test_ds, logger, args)
        print(results)
