import torch
import torch.nn as nn
from torchvision.transforms import v2

class Augment_v2(nn.Module):
    """
    MoCo v2 augments
    Adapted from:
        Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, Ross Girshick; 
        "Momentum Contrast for Unsupervised Visual Representation Learning";
        https://github.com/facebookresearch/moco
    """
    def __init__(self, image_size):
        super().__init__()
        
        # Compute kernel size for blur
        kernel_size = int(0.1*image_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Define augment
        self.augment = v2.Compose([
            v2.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0))], p=0.5),
            v2.RandomHorizontalFlip(),
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
            # Cifar10 normalization
            v2.Normalize(mean=(0.4914, 0.4822, 0.4465), 
                         std=(0.2023, 0.1994, 0.2010))
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
