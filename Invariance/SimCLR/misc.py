import torch
import torch.nn as nn
from torchvision.transforms import v2

class Augment(nn.Module):
    """
    SimCLR augments
    Adapted from:
        Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
        "A Simple Framework for Contrastive Learning of Visual Representations";
        https://github.com/google-research/simclr
    """
    def __init__(self, image_size, s = 1):
        super().__init__()
        
        # Compute kernel size for blur
        kernel_size = int(0.1*image_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Define augment
        self.augment = v2.Compose([
            v2.RandomResizedCrop(224),
            v2.RandomHorizontalFlip(),
            v2.RandomApply([v2.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)], p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0))], p=0.5),
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, x):
        return self.augment(x)

def collate_fn(batch, augment):
    """
    batch: list of (img, label)
    augment: an instance of Augment_v2
    """
    x_q, x_k = [], []
    for img, label in batch:
        x1 = augment(img)
        x2 = augment(img)
        x_q.append(x1)
        x_k.append(x2)

    # Stack into tensors
    x_q = torch.stack(x_q, dim=0)
    x_k = torch.stack(x_k, dim=0)

    return x_q, x_k
