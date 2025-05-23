import argparse

import torch
import torchvision
from torch.utils.data import DataLoader

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from misc import collate_fn, Augment
from model import MAE

def parse_arguments():
    """Parse command line arguments for model training."""
    parser = argparse.ArgumentParser(description="Training script")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_224", 
                        help="Name of the model architecture.")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=200, 
                        help="Total training epochs.")
    parser.add_argument("--warmup_epochs", type=int, default=10, 
                        help="Number of warmup epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, 
                        help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-6, 
                        help="Weight decay.")
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.add_argument('--patch_size', default=4, type=int,
                        help='Patch size in pixels.')
    parser.add_argument('--in_chans', default=3, type=int,
                        help='Number of channels in images.')
    
    # Batch size configuration
    parser.add_argument("--batch_size", type=int, default=256, 
                        help="Per-device batch size.")
    parser.add_argument("--full_batch_size", type=int, default=1024, 
                        help="Target effective batch size.")
    parser.add_argument("--accum_grad_steps", type=int, default=1, 
                        help="Steps to accumulate gradients.")
    
    # Hardware configuration
    parser.add_argument("--gpu_id", type=int, nargs="+", default=[0],
                        help="GPU ids to use.")
    parser.add_argument('--nodes', type=int, default=1, 
                        help='Number of compute nodes.')
    parser.add_argument('--num_workers', type=int, default=20, 
                        help='Number of data loading workers.')
    parser.add_argument("--distributed", action="store_true", 
                        help="Use distributed training.")
    
    return parser.parse_args()

def setup(args):
    """Calculate derived parameters based on input arguments."""
    # Calculate effective batch size across all devices
    eff_batch_size = args.batch_size * len(args.gpu_id) * args.nodes
    
    # Calculate gradient accumulation steps needed
    accum_grad_steps = args.full_batch_size // eff_batch_size
    
    # Use linear scaling rule
    args.lr = args.lr * args.full_batch_size / 256
    args.accum_grad_steps = accum_grad_steps
    
    return args

def main(args):
    """Main training function."""
    # Dataset setup
    train_dataset = torchvision.datasets.CIFAR10(root="../../data/", train=True)
    img_size = train_dataset[0][0].size[0]
    
    # Data loader with augmentation
    augment = Augment(image_size=img_size)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, augment),
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Logger and checkpointing
    logger = TensorBoardLogger("logs", name=args.model_name)
    checkpoint_callback = ModelCheckpoint(
        filename="model--{epoch}-{train_loss:.2f}",
        monitor="train_loss",
        save_top_k=1,
        mode="min"
    )
    
    # Set numerical precision
    torch.set_float32_matmul_precision("medium")
    
    # Trainer
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.gpu_id,
        num_nodes=args.nodes,
        strategy="ddp" if args.distributed else "auto",
        accumulate_grad_batches=args.accum_grad_steps,
        log_every_n_steps=1,
        logger=logger,
        callbacks=[checkpoint_callback]
    )
    
    # Initialize model
    model = MAE(args.model_name, 
                img_size, 
                args.epochs, 
                args.warmup_epochs, 
                args.weight_decay, 
                args.lr, 
                norm_pix_loss=args.norm_pix_loss,
                mask_ratio=args.mask_ratio,
                patch_size=args.patch_size,
                in_chans=args.in_chans
                )

    # Fit the model
    trainer.fit(model, train_dataloaders=train_loader)

if __name__ == "__main__":
    args = parse_arguments()
    args = setup(args)
    main(args)