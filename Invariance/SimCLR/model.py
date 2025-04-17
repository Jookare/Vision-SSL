import torch
import lightning.pytorch as pl
import timm
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

class SimCLR(pl.LightningModule):
    """
    SimCLR as Pytorch LightningModule. The variables are utilized from the self.hparams
    """
    def __init__(self, model_name, img_size, epochs, warmup_epochs, weight_decay, lr, tau, mlp_dim=4096, proj_dim=128):
        super().__init__()
        self.save_hyperparameters()

        # Backbone encoder
        self.encoder = timm.create_model(model_name, pretrained=False, num_classes=0, img_size=img_size)
        embed_dim = self.encoder.num_features
        self.encoder.head = nn.Linear(embed_dim, embed_dim)
        self.batch_norm = nn.BatchNorm1d(embed_dim)

        # MLP projection head
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, proj_dim, bias=False),
        )

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, x):
        features = self.encoder(x)
        return features
    
    def training_step(self, batch, batch_idx):
        x1, x2 = batch[0], batch[1]  # Two views of the same batch
        
        z1 = self.projector(self.batch_norm(self.forward(x1)))
        z2 = self.projector(self.batch_norm(self.forward(x2)))
        
        # Symmetric loss
        loss = 0.5*(self.InfoNCE_loss(z1, z2) + self.InfoNCE_loss(z2, z1))
        
        # Batch size
        N = x1.shape[0]
        self.log("train_loss", loss, batch_size=N)
        return loss
    
        
    def InfoNCE_loss(self, z1, z2):
        """
        Computes the InfoNCE loss between two batches of projected features.

        This implementation avoids the traditional 2N-based formulation of SimCLR by treating 
        the two augmented views (z1 and z2) as separate inputs. Instead of concatenating them, 
        we compute the cosine similarity matrix directly between z1 and z2, and use the 
        diagonal as the positive pairs. Each row in z1 is matched with the same index in z2.

        This maximizes the similarity between corresponding (positive) pairs while 
        minimizing it with respect to the rest (negatives in the same batch).
        """
        # Compute cosine similarity
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)

        # Compute similarity matrix
        logits = torch.matmul(z1, z2.T) / self.hparams.tau
        
        # Batch size
        N = logits.shape[0]
        
        # The diagonal should be the largest element
        labels = torch.arange(N, device=logits.device)
        loss = self.loss_fn(logits, labels)
        
        return loss
        
    def configure_optimizers(self):
        params = list(self.encoder.parameters()) + list(self.projector.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.hparams.lr)
        
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