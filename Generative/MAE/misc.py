import torch
import torch.nn as nn
from torchvision.transforms import v2

class Augment(nn.Module):
    """
    MAE augmentations from the paper
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

def collate_fn(batch, augment):
    """
    batch: list of (img, label)
    augment: an instance of Augment
    """
    x = []
    for img, label in batch:
        x1 = augment(img)
        x.append(x1)

    # Stack into tensors
    x = torch.stack(x, dim=0)
    return x
