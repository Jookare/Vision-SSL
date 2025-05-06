import torch
import lightning.pytorch as pl
import timm
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

class IJEPA(pl.LightningModule):
    """
    I-JEPA implemented as a PyTorch LightningModule.
    """
    def __init__(self, model_name, img_size, epochs, warmup_epochs, weight_decay, m, lr, mlp_dim=2048, proj_dim=384):
        super().__init__()
        self.save_hyperparameters()

        # Context encoder
        self.context_encoder = timm.create_model(model_name, pretrained=False, num_classes=0, img_size=img_size)
        embed_dim = self.context_encoder.num_features
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, proj_dim),
        )
        
        # Target encoder
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self._freeze_target_network()
    
    def _freeze_target_network(self):
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.context_encoder(x)
    
    def forward_loss(self, batch):
        imgs, ctx_mask, tgt_mask = batch  # [B, C, H, W], [B, 1, N], [B, n_tgt, N]
        B, n_tgt, N = tgt_mask.shape

        # Get context features
        ctx_feat = self.context_encoder(imgs)          # [B, N, D]
        ctx_proj = self.predictor(ctx_feat)            # [B, N, D_proj]

        # Get target features (frozen)
        with torch.no_grad():
            tgt_feat = self.target_encoder(imgs)       # [B, N, D_proj]

        ctx_out, tgt_out = [], []

        for i in range(B):
            for j in range(n_tgt):
                indices = tgt_mask[i, j]
                ctx_out.append(ctx_proj[i, indices])
                tgt_out.append(tgt_feat[i, indices])

        pred = torch.cat(ctx_out, dim=0)
        target = torch.cat(tgt_out, dim=0)

        loss = F.mse_loss(pred, target)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.forward_loss(batch)
        self._momentum_update()
        self.log("train_loss", loss)
        return loss
        
    @torch.no_grad()
    def _momentum_update(self):
        """EMA update of the target network parameters."""
        m = self.hparams.m
        for context_param, target_param in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            target_param.data = m * target_param.data + (1 - m) * context_param.data
            
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        
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