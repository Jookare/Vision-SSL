import torch
import torch.nn as nn
from torchvision.transforms import v2
import torch.nn as nn
import math
from multiprocessing import Value

class Augment(nn.Module):
    """
    I-JEPA augmentations from the paper
    """
    def __init__(self, image_size, s = 1):
        super().__init__()
        
        # Compute kernel size for blur
        kernel_size = int(0.1*image_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Define augment
        self.augment = v2.Compose([
            v2.RandomResizedCrop(image_size, scale=(0.2, 1.0), interpolation=3), # 3 is bicubic
            v2.RandomHorizontalFlip(),
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
            # Cifar10 normalization
            v2.Normalize(mean=(0.4914, 0.4822, 0.4465), 
                         std=(0.2023, 0.1994, 0.2010))
        ])

    def forward(self, x):
        return self.augment(x)

class Multiblock_masking(nn.Module):
    
    def __init__(self, 
                 img_size=(224, 224), 
                 patch_size=16, 
                 ctx_scale=(0.85, 1.0),
                 tgt_scale=(0.15, 0.2),
                 tgt_aspect_ratio=(0.75, 1.5),
                 allow_overlap=False,
                 shared_masks=False):
        super().__init__()
        self.patch_size = patch_size
        self.height = img_size[0] // patch_size  # in patches
        self.width = img_size[1] // patch_size   # in patches
        self.ctx_scale = ctx_scale
        self.tgt_scale = tgt_scale
        self.tgt_aspect_ratio = tgt_aspect_ratio
        self.allow_overlap = allow_overlap
        self.n_tgt = 4
        self.min_keep = 4 # Minimum mask size
        self.shared_masks=shared_masks
        self._itr_counter = Value('i', -1)  # collator is shared across worker processes

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v
    
    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        # Sample random number from uniform
        _rand = torch.rand(1, generator=generator).item()
        
        # Define block size 
        min_s, max_s = scale
        mask_scale = min_s + _rand*(max_s - min_s)
        N_patches = int(self.height * self.width * mask_scale)
        
        # Define aspect ratio
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand*(max_ar - min_ar)
        
        # Compute the block height and width in patches
        h = int(round(math.sqrt(N_patches * aspect_ratio)))
        w = int(round(math.sqrt(N_patches / aspect_ratio)))

        assert h <= self.width and w <= self.width, "Block height and width should be less than image size"

        return (h, w)
    
    def _sample_block_mask(self, block_size):
        h, w = block_size
        
        # Limit to 100 tries to avoid infinite loops
        for _ in range(100): 
            # Sample block top-left corner
            top = 0 if h == self.height else torch.randint(0, self.height - h + 1, (1,)).item()
            left = 0 if w == self.width else torch.randint(0, self.width - w + 1, (1,)).item()

            # Define 2d mask
            mask = torch.zeros((self.height, self.width), dtype=torch.bool)
            mask[top:top+h, left:left+w] = 1
    
            if mask.sum() >= self.min_keep:
                return mask

        raise RuntimeError("Failed to sample a non-overlapping block after 100 attempts")
    
    def _mask_to_indices(self, mask):
        return torch.nonzero(mask.flatten(), as_tuple=False).squeeze(1)
        
    def forward(self, batch):
        # batch is list of tuples
        B = len(batch)
        
        # Stack images to a single tensor
        collated_batch = torch.stack([B[0] for B in batch], dim=0)
        seed = self.step()
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        # Sample sizes
        tgt_size = self._sample_block_size(generator, self.tgt_scale, self.tgt_aspect_ratio)
        ctx_size = self._sample_block_size(generator, self.ctx_scale, (1.0, 1.0))
        
        if self.shared_masks:
            tgt_blocks = []
            tgt_mask_union = torch.zeros((self.height, self.width), dtype=torch.bool)

            for _ in range(self.n_tgt):
                tgt_mask = self._sample_block_mask(tgt_size)
                tgt_blocks.append(self._mask_to_indices(tgt_mask))
                tgt_mask_union |= tgt_mask

            ctx_mask = self._sample_block_mask(ctx_size)
            ctx_mask = ctx_mask & ~tgt_mask_union
            ctx_idx = self._mask_to_indices(ctx_mask)

            tgt_mask_tensor = torch.stack(tgt_blocks)
            ctx_mask_tensor = ctx_idx.unsqueeze(0)

            # Expand the masks for all image in the batch
            collated_masks_tgt = tgt_mask_tensor.unsqueeze(1).repeat(1, B,  1)
            collated_masks_ctx = ctx_mask_tensor.unsqueeze(1).repeat(1,B,  1)  

        else:
            # Finding unique mask for all image
            tgt_size = self._sample_block_size(generator, self.tgt_scale, self.tgt_aspect_ratio)
            ctx_size = self._sample_block_size(generator, self.ctx_scale, (1.0, 1.0))

            ctx_indices = []
            tgt_indices = []
            min_keep_tgt = self.height * self.width
            min_keep_ctx = self.height * self.width

            for _ in range(B):
                tgt_blocks = []
                tgt_mask_union = torch.zeros((self.height, self.width), dtype=torch.bool)

                for _ in range(self.n_tgt):
                    tgt_mask = self._sample_block_mask(tgt_size)
                    tgt_blocks.append(self._mask_to_indices(tgt_mask))
                    min_keep_tgt = min(min_keep_tgt, tgt_mask.sum())
                    tgt_mask_union |= tgt_mask

                ctx_mask = self._sample_block_mask(ctx_size)
                ctx_mask = ctx_mask & ~tgt_mask_union
                ctx_idx = self._mask_to_indices(ctx_mask)
                min_keep_ctx = min(min_keep_ctx, ctx_mask.sum())

                tgt_indices.append(tgt_blocks)
                ctx_indices.append([ctx_idx])

            # Stack the masks along first dimension
            collated_masks_tgt = [
                torch.stack([block[:min_keep_tgt] for block in tgt_mask], dim=0)
                for tgt_mask in tgt_indices
            ]
            collated_masks_ctx = [
                torch.stack([block[:min_keep_ctx] for block in ctx_mask], dim=0)
                for ctx_mask in ctx_indices
            ]
            

            collated_masks_tgt = torch.stack(collated_masks_tgt, dim=1)  # [n_tgt, B, min_keep_tgt]
            collated_masks_ctx = torch.stack(collated_masks_ctx, dim=1)  # [1, B, min_keep_ctx]
            
        return collated_batch, collated_masks_ctx, collated_masks_tgt