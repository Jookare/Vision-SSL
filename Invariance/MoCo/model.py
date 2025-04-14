import torch
import lightning.pytorch as pl
import timm
import copy
import torch.nn as nn
from torchvision.transforms import v2


class MoCo(pl.LightningModule):
    def __init__(self, model_name, m=0.99, tau=0.07, lr=1e-3, mlp_dim=4096, pred_dim=256):
        super().__init__()
        self.save_hyperparameters()
        self.tau = tau
        self.m = m
        
        # Backbone (feature extractor)
        backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        embed_dim = backbone.num_features
        
        # Online encoder (f_q): backbone + projection + prediction as in MoCo v3
        self.encoder = backbone
        self.encoder.head = self._build_mlp(3, embed_dim, mlp_dim, pred_dim)
        self.predictor = self._build_mlp(2, pred_dim, mlp_dim, pred_dim)

        # Offline encoder (f_k): backbone + projection
        self.momentum_encoder = copy.deepcopy(self.encoder)
        
        self.loss_fn = nn.CrossEntropyLoss()
        # Remove grad from momentum_encoder
        for param in self.momentum_encoder.parameters():
            param.requires_grad = False
    
    
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
        self.log("train_loss", loss)
        return loss
    
        
    def contrastive_loss(self, q, k):
        # Normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        logits = torch.matmul(q, k.T) / self.tau
        print("logits shape:", logits.shape)
        
        # Batch size
        N = logits.shape[0]
        print("N:", N)
        
        # The diagonal should be the largest element
        labels = torch.arange(N, device=logits.device)
        loss = self.loss_fn(logits, labels)
        print("Loss:", loss)
        
        return 2 * self.tau * loss
        
    # def validation_step(self, ):
        
    #     return ...
    
    @torch.no_grad()
    def _momentum_update(self):
        for param_q, param_k in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
            param_k.data = self.m * param_k.data + (1. - self.m) * param_q.data

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.encoder.parameters(), lr=self.hparams.lr)
        return optim