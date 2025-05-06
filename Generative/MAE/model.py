import torch
import lightning.pytorch as pl
import timm
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from timm.models.vision_transformer import PatchEmbed, Block

class MAE(pl.LightningModule):
    """
    MAE implemented as a PyTorch LightningModule.
    """
    def __init__(self, model_name, img_size, epochs, warmup_epochs, weight_decay, lr, norm_pix_loss=True, 
                 decoder_dim=512, patch_size=16, mask_ratio=0.75, in_chans = 3):
        super().__init__()
        self.save_hyperparameters()

        # Initialize backbone Vision Transformer from timm
        self.encoder = timm.create_model(model_name, pretrained=False, num_classes=0, img_size=img_size)
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.img_size = img_size
        self.norm_pix_loss = norm_pix_loss
        
        # Define encoder dimensions
        self.encoder_dim = self.encoder.num_features
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding layer
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, self.encoder_dim)
        
        # Position embeddings and cls_token for encoder
        self.encoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.encoder_dim))
        nn.init.trunc_normal_(self.encoder_pos_embed, std=0.02)
        
        # Decoder
        self.decoder = MAE_decoder(
            in_chans=in_chans,
            patch_size=patch_size,
            num_patches=self.num_patches,
            encoder_dim=self.encoder_dim,
            decoder_embed_dim=decoder_dim
        )
        
        assert hasattr(self.encoder, "blocks"), "encoder should have blocks "

    def forward(self, imgs):
        """Forward pass for downstream tasks"""
        # Convert images to patches
        x = self.patch_embed(imgs)
        
        # Add positional embeddings
        x = x + self.encoder_pos_embed
        
        # Pass through encoder blocks
        x = self.encoder.blocks(x)
        x = self.encoder.norm(x)

        # Mean pooling over the patches as cls_token is not actually required in ViT
        # https://github.com/google-research/vision_transformer/issues/61#issuecomment-802233921
        x = x.mean(dim=1)
        
        return x
    
    def forward_loss(self, imgs):
        """Forward pass with loss calculation for pretraining"""
        # Get patches as reconstruction targets
        target = self.patchify(imgs)

        # Normalize pixels if needed
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        
        # Encode with masking
        x_encoded, mask, ids_restore = self.encode(imgs)
        
        # Decode and predict pixel values
        pred = self.decoder(x_encoded, ids_restore)
        
        # Calculate MSE loss only on masked patches
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        
        return loss, pred, mask

    def training_step(self, batch, batch_idx):
        imgs = batch
        loss, _, _ = self.forward_loss(imgs)
        
        self.log("train_loss", loss, batch_size=imgs.size(0))
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9, nesterov=True)
        
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

        
    def patchify(self, imgs):
        """
        Convert images to patches
        # [B, C, H, W] -> [B, C, H/P, P, W/P, P] -> [B, N, D]
        """
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % self.patch_size == 0
        
        p = self.patch_size
        h = w = imgs.shape[2] // p
        c = self.hparams.in_chans
        
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        
        # Permute the tensor
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def random_masking(self, x):
        """Randomly masks patches"""
        # Find number of patches and number to keep
        N = x.shape[1]
        len_keep = int(N * (1 - self.mask_ratio))

        # Use random noise from U[0,1] to select random patches
        noise = torch.rand(x.shape[0], N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Select the first len_keep patch indices
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[2]))

        # Create a binary mask, where 1 = masked, 0 = visible
        mask = torch.ones([x.shape[0], N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def encode(self, imgs):
        """Encode images with masking - for pretraining"""
        
        # Convert images to patches
        x = self.patch_embed(imgs)
        
        # Add positional embeddings
        x = x + self.encoder_pos_embed
        
        # Apply random masking
        x_masked, mask, ids_restore = self.random_masking(x)
        
        # Pass through encoder blocks
        x_masked = self.encoder.blocks(x_masked)
        x_encoded = self.encoder.norm(x_masked)
        
        return x_encoded, mask, ids_restore
    
    


class MAE_decoder(nn.Module):
    """
    Masked Autoencoder Decoder Module
    
    This decoder reconstructs the original image from encoded visible patches.
    It inserts mask tokens for missing patches and predicts their pixel values.
    """
    def __init__(
        self,
        in_chans: int = 3,
        patch_size: int = 16,
        num_patches: int = 196,
        encoder_dim: int = 1024,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
    ):
        super().__init__()
        # Project from encoder dimension to decoder dimension
        self.decoder_embed = nn.Linear(encoder_dim, decoder_embed_dim, bias=False)

        # Learnable mask token and position embeddings
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim))

        # Transformer decoder blocks
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=decoder_embed_dim,
                    num_heads=decoder_num_heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    norm_layer=lambda x : nn.LayerNorm(x, eps=1e-6)
                )
                for _ in range(decoder_depth)
            ]
        )

        # Final normalization and prediction head
        self.norm = nn.LayerNorm(decoder_embed_dim)
        self.head = nn.Linear(decoder_embed_dim, in_chans * patch_size ** 2)

        self._init_weights()

    def _init_weights(self):
        """Initialize position embeddings and mask token with truncated normal distribution"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, input, ids_restore):
        """
        Decode the visible patches and predict the masked patches
        Args:
            encoded_visible: [B, N_vis, encoder_dim] - Encoded visible patches
            ids_restore: [B, N] - Mapping to restore original patch positions
        
        Returns:
            pred: [B, N_total, patch_dim] - Predicted pixel values for all patches
        """
        B, _, C = input.shape  # B = batch size, C = encoder dimension
        
        # Project encoder features to decoder dimension
        x = self.decoder_embed(input)
        
        # Create mask tokens for all masked positions
        mask_tokens = self.mask_token.repeat(B, ids_restore.shape[1] - x.shape[1], 1)

        # Concatenate visible tokens with mask tokens [B, N, decoder_dim]
        x_ = torch.cat([x, mask_tokens], dim=1)  

        # Unshuffle: restore patches to original positions
        x_ = torch.gather(
            x_, dim=1, 
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2])
        )
        
        # Add position embeddings
        x = x_ + self.pos_embed
        
        # Apply decoder transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        
        # Project to pixel values
        pred = self.head(x) # [B, N, P*P*C]
        return pred
    
    