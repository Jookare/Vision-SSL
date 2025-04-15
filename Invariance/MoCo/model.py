import torch
import lightning.pytorch as pl
import timm
import copy
import torch.nn as nn
from torchvision.transforms import v2
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

class MoCo(pl.LightningModule):
    """
    Momentum Contrast as Pytorch LightningModule. The variables are utilized from the self.hparams
    """
    def __init__(self, model_name, epochs, warmup_epochs, m=0.99, tau=0.2, lr=1e-4, mlp_dim=4096, pred_dim=256):
        super().__init__()
        self.save_hyperparameters()
        
        # Backbone (feature extractor)
        backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        embed_dim = backbone.num_features
        
        # Online encoder (f_q): backbone + projection + prediction as in MoCo v3
        self.encoder = backbone
        self.encoder.head = self._build_mlp(3, embed_dim, mlp_dim, pred_dim)
        self.predictor = self._build_mlp(2, pred_dim, mlp_dim, pred_dim)

        # Offline encoder (f_k): backbone + projection
        self.momentum_encoder = copy.deepcopy(self.encoder)
        for param in self.momentum_encoder.parameters():
            param.requires_grad = False

        self.loss_fn = nn.CrossEntropyLoss()
    
    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        """
        Adapted from https://github.com/facebookresearch/moco-v3
        """
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)
        
    def training_step(self, batch, batch_idx):
        x1, x2 = batch[0], batch[1]
        
        # MoCo v3 query/key
        q1 = self.encoder(x1)
        q2 = self.encoder(x2)
        with torch.no_grad():
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)
        
        # Symmetric loss
        loss = self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)
        self._momentum_update()
        
        # Batch size
        N = x1.shape[0]
        self.log("train_loss", loss, batch_size=N)
        return loss
    
        
    def contrastive_loss(self, q, k):
        # Normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        logits = torch.matmul(q, k.T) / self.hparams.tau
        
        # Batch size
        N = logits.shape[0]
        
        # The diagonal should be the largest element
        labels = torch.arange(N, device=logits.device)
        loss = self.loss_fn(logits, labels)
        
        return 2 * self.hparams.tau * loss
        
    @torch.no_grad()
    def _momentum_update(self):
        for param_q, param_k in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
            param_k.data = self.hparams.m * param_k.data + (1. - self.hparams.m) * param_q.data

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=self.hparams.lr)
        
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