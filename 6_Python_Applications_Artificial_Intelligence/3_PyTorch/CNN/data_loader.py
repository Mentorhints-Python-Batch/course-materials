"""Data loading and preprocessing utilities for Intel Image Classification."""

import os

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
NUM_CLASSES = len(CLASS_NAMES)

IMAGE_SIZE = 224  # ResNet expects 224x224
ORIGINAL_SIZE = 150
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_BASE_DIR = os.path.dirname(__file__)
TRAIN_DIR = os.path.join(_BASE_DIR, "Dataset", "seg_train", "seg_train")
TEST_DIR = os.path.join(_BASE_DIR, "Dataset", "seg_test", "seg_test")
PRED_DIR = os.path.join(_BASE_DIR, "Dataset", "seg_pred", "seg_pred")


def get_train_transforms(image_size=IMAGE_SIZE):
    """Transforms for training data with augmentation."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_test_transforms(image_size=IMAGE_SIZE):
    """Transforms for validation / test data (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_datasets(train_dir=TRAIN_DIR, test_dir=TEST_DIR, image_size=IMAGE_SIZE):
    """Load train and test datasets using ``ImageFolder``.

    Returns
    -------
    train_dataset, test_dataset : torchvision.datasets.ImageFolder
    """
    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=get_train_transforms(image_size),
    )
    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=get_test_transforms(image_size),
    )
    return train_dataset, test_dataset


def split_train_val(train_dataset, val_ratio=0.15, seed=42):
    """Split a training dataset into train and validation subsets.

    Returns
    -------
    train_subset, val_subset : torch.utils.data.Subset
    """
    total = len(train_dataset)
    val_size = int(total * val_ratio)
    train_size = total - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(train_dataset, [train_size, val_size], generator=generator)


def create_dataloaders(train_subset, val_subset, test_dataset,
                       batch_size=32, num_workers=2):
    """Create DataLoader instances for train, val, and test splits.

    Returns
    -------
    train_loader, val_loader, test_loader : torch.utils.data.DataLoader
    """
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Reverse ImageNet normalization for display purposes.

    Parameters
    ----------
    tensor : torch.Tensor of shape (C, H, W) or (B, C, H, W)
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    return (tensor * std + mean).clamp(0, 1)
