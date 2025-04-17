import torch
import lightning.pytorch as pl
import timm
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

class BarlowTwins(pl.LightningModule):
    """
    Barlow Twins implemented as a PyTorch LightningModule.
    """
    def __init__(self, model_name, img_size, epochs, warmup_epochs, weight_decay, tau, lr, mlp_dim=2048):
        super().__init__()
        self.save_hyperparameters()

        # encoder
        self.encoder = timm.create_model(model_name, pretrained=False, num_classes=0, img_size=img_size)
        embed_dim = self.encoder.num_features

        # Online predictor
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
        )
        
    def forward(self, x):
        return self.encoder(x)
    
    def training_step(self, batch, batch_idx):
        x1, x2 = batch[0], batch[1]
    
        # Projections
        z1 = self.projector(self.forward(x1))
        z2 = self.projector(self.forward(x2))
        
        # Compute loss
        loss = self.BT_loss(z1, z2)
        self.log("train_loss", loss, batch_size=x1.size(0))
        return loss
    
    def BT_loss(self, z1, z2):
        N, D = z1.shape
        
        # Normalize each feature to same scale
        z1 = (z1 - z1.mean(0)) / z1.std(0)
        z2 = (z2 - z2.mean(0)) / z2.std(0)

        # Cross-correlation matrix
        C = torch.mm(z1.T, z2) / N

        # Loss
        on_diag = torch.diagonal(C).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(C).pow_(2).sum()
        loss = on_diag + self.hparams.tau * off_diag
        return loss

    def off_diagonal(self, x):
        # Returns a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        
        # Flatten, remove last element and reshape to (n-1, n+1)
        # e.g., If n == m == 10, the shape will be (9, 11) and the diagonal 
        # values are now in the first column.
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
        
    def configure_optimizers(self):
        params = (
            list(self.encoder.parameters()) +
            list(self.projector.parameters())
        )

        optimizer = torch.optim.SGD(params, lr=self.hparams.lr, momentum=0.9, nesterov=True)
        
        # Create linear warmup scheduler
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.hparams.warmup_epochs
        )

        # Create cosine decay scheduler
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.epochs - self.hparams.warmup_epochs,
            eta_min=0.0
        )

        # Combine them in sequence
        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.hparams.warmup_epochs]
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}