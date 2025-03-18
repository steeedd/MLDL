
import os
import shutil

from torchvision.datasets import ImageFolder
import torchvision.transforms as T

import torch
from torch.utils.data import DataLoader

from torch import nn

from collections import Counter

def getdata():
    # Transformations to Apply on EACH Input Image
    transform = T.Compose([
        T.Resize((224, 224)),  # Resize to Fit the Input Dimensions of the Network
        T.ToTensor(), # Tensor (Channels=3 x Height=224 x Width = 224)
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

        # Create the Training Dataset
    tiny_imagenet_dataset_train = ImageFolder(root='tiny-imagenet/tiny-imagenet-200/train', transform=transform)

    # Create the Validation Dataset
    tiny_imagenet_dataset_val = ImageFolder(root='tiny-imagenet/tiny-imagenet-200/val', transform=transform)



    # Create the DataLoader for the Training:
    train_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_train, batch_size=32, shuffle=True, num_workers=4)

    # Create the DataLoader for the Validation
    val_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_val, batch_size=32, shuffle=False)

    return train_loader, val_loader
