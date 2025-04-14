import torch
import lightning.pytorch as pl
import timm

class MoCo(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.encoder = timm.create_model(backbone)
        