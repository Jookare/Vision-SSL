import torch
import lightning.pytorch as pl
import timm
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

class BYOL(pl.LightningModule):
    """
    BYOL implemented as a PyTorch LightningModule.
    """
    def __init__(self, model_name, img_size, epochs, warmup_epochs, weight_decay, m, tau, lr, mlp_dim=4096, proj_dim=256):
        super().__init__()
        self.save_hyperparameters()

        # Online encoder
        self.online_encoder = timm.create_model(model_name, pretrained=False, num_classes=0, img_size=img_size)
        embed_dim = self.encoder.num_features

        # Online projector
        self.online_projector = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, proj_dim),
        )

        # Online predictor
        self.online_predictor = nn.Sequential(
            nn.Linear(proj_dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, proj_dim),
        )
        
        # Target encoder and projector (no predictor)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        
        # Freeze target network parameters
        self._freeze_target_network()
    
    def _freeze_target_network(self):
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.online_encoder(x)
    
    def target_forward(self, x):
        """
        Forward pass through target encoder + projector.
        """
        y = self.target_encoder(x)
        z = self.target_projector(y)
        return z
    
    def training_step(self, batch, batch_idx):
        x1, x2 = batch[0], batch[1]
    
        # Online network predictions
        y1 = self.forward(x1)
        y2 = self.forward(x2)
        q1 = self.online_predictor(self.online_projector(y1))
        q2 = self.online_predictor(self.online_projector(y2))
        
        # Target network projections (stop gradient)
        with torch.no_grad():
            z1 = self.target_forward(x1)
            z2 = self.target_forward(x2)
        
        # Compute BYOL loss
        N = x1.shape[0]
        loss = self.byol_loss(q1, z2) + self.byol_loss(q2, z1)
        self._momentum_update()
        self.log("train_loss", loss, batch_size=x1.size(0))
        return loss
    
    def byol_loss(self, p, z):
        # Normalize
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)

        # Compute cosine similarity across features and scale losses to range [0, 4]
        return 2 - 2 * (p * z).sum(dim=1).mean()
        
    @torch.no_grad()
    def _momentum_update(self):
        """EMA update of the target network parameters."""
        m = self.hparams.m
        for online_param, target_param in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_param.data = m * target_param.data + (1 - m) * online_param.data
            
        for online_param, target_param in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            target_param.data = m * target_param.data + (1 - m) * online_param.data

    def configure_optimizers(self):
        params = (
            list(self.online_encoder.parameters()) +
            list(self.online_projector.parameters()) +
            list(self.online_predictor.parameters())
        )

        optimizer = torch.optim.AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        
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