import torch
import lightning.pytorch as pl
import timm
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from timm.models.vision_transformer import Block, PatchEmbed

class IJEPA(pl.LightningModule):
    """
    I-JEPA implemented as a PyTorch LightningModule.
    """
    def __init__(self, model_name, img_size, patch_size, epochs, warmup_epochs, weight_decay, m, lr, predictor_depth=6):
        super().__init__()
        self.save_hyperparameters()

        # Context encoder
        self.context_encoder = IJEPA_encoder(img_size=img_size, patch_size=patch_size, num_heads=12)
        
        # Predictor
        self.predictor = IJEPA_predictor(
            num_patches=(img_size//patch_size)**2,
            embed_dim=self.context_encoder.get_output_dim(),
            depth=predictor_depth,
            num_heads=12
        )
        
        # Target encoder
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self._freeze_target_network()
    
    def _freeze_target_network(self):
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.context_encoder(x)

    def training_step(self, batch, batch_idx):
        images, masks_context, masks_target = batch
        z = self.context_encoder(images, masks_context)
        z = self.predictor(z, masks_context, masks_target)
        
        h = self.target_encoder(images)
        h = apply_masks(h, masks_target)
        
        loss = (h - z)**2
        loss = loss.mean(dim=-1)
        
        self._momentum_update()
        self.log("train_loss", loss, batch_size=images.shape[0])
        return loss
        
    @torch.no_grad()
    def _momentum_update(self):
        """EMA update of the target network parameters."""
        m = self.hparams.m
        for context_param, target_param in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            target_param.data = m * target_param.data + (1 - m) * context_param.data
            
    def configure_optimizers(self):
        params = (
            list(self.context_encoder.parameters())+
            list(self.predictor.parameters())
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


class IJEPA_encoder(nn.Module):
    """
    Custom ViT encoder for I-JEPA.
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, masks=None):
        
        # patchify x
        x = self.patch_embed(x)
        B, N, D = x.shape
        
        x = x + self.pos_embed
        
        if masks is not None:
            x = apply_masks(x, masks)
            
        x = self.blocks(x)
        x = self.norm(x)
        
        return x

    def get_output_dim(self):
        return self.patch_embed.proj.out_channels


class IJEPA_predictor(nn.Module):
    """
    Lightweight ViT predictor for I-JEPA as described in the paper.
    The predictor maps context embeddings to predicted target embeddings.
    Uses a narrower ViT architecture (fixed embedding dim of 384).
    """
    def __init__(
        self,
        num_patches,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,  # For ViT-B/16; use 12 for ViT-L/16, ViT-H/16, ViT-H/14; use 16 for ViT-G/16
        num_heads=12,  # Should match the number of heads in context encoder
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        
        # Input projection to predictor embedding dimension
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[
            Block(
                dim=predictor_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])
        
        self.norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim)
        
    
    def forward(self, x, masks_context, masks_target):
        assert (masks_context is not None) and (masks_target is not None), 'Cannot run predictor without mask indices'
        
        # Determine batch size
        B = len(x) // len(masks_context)
        
        # Project from encoder_dim to predictor_dim
        x = self.predictor_embed(x)
        
        # Add positional embeddings to context tokens
        x_pos_embed = self.pos_embed.repeat(B, 1, 1)
        x += apply_masks(x_pos_embed, masks_context)
        
        _, N_ctxt, D = x.shape
        
        # Find positional embeddings for target tokens
        pos_embs = self.pos_embed.repeat(B, 1, 1)
        pos_embs = apply_masks(pos_embs, masks_target)
        
        # Init masked tokens and add positional embedding
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        pred_tokens += pos_embs
        
        # Combine Context and Mask Tokens
        x = x.repeat(len(masks_target), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)
        
        # Pass through transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        
        # Return the predictions
        x = x[:, N_ctxt:]
        
        # Project back to output dimension
        x = self.predictor_proj(x)
        
        return x

def apply_masks(x, masks):
    """
    Safer version: applies masks to x, with error checks.
    """
    all_x = []
    for i, m in enumerate(masks):
        if m.max().item() >= x.size(1):
            raise ValueError(f"Mask index {m.max().item()} out of bounds for input of size {x.size(1)} (batch {i})")
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        selected = torch.gather(x, dim=1, index=mask_keep)
        all_x.append(selected)
    return torch.cat(all_x, dim=0)