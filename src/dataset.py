from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def get_transforms(use_augmentation: bool):
    """
    Train: optional random augmentation.
    Val/Test: deterministic preprocessing only.
    """
    normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    if use_augmentation:
        train_transform = transforms.Compose(
            [
                transforms.Resize(110),
                transforms.RandomCrop(96),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize(110),
                transforms.CenterCrop(96),
                transforms.ToTensor(),
                normalize,
            ]
        )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(110),
            transforms.CenterCrop(96),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


def build_stl10_datasets(data_root: str, use_augmentation: bool):
    """
    Load course-local STL10 subset from:
      data_root/train/<class_name>/*
      data_root/test/<class_name>/*
    """
    root = Path(data_root)
    train_dir = root / "train"
    test_dir = root / "test"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Expected local dataset folders not found: {train_dir} and {test_dir}"
        )

    train_transform, eval_transform = get_transforms(use_augmentation=use_augmentation)

    # Two train datasets share same files but use different transforms.
    train_set = datasets.ImageFolder(root=str(train_dir), transform=train_transform)
    val_set = datasets.ImageFolder(root=str(train_dir), transform=eval_transform)
    test_set = datasets.ImageFolder(root=str(test_dir), transform=eval_transform)
    return train_set, val_set, test_set


def build_dataloaders(
    train_set,
    val_set,
    test_set,
    val_ratio=0.1,
    batch_size=128,
    num_workers=0,
    seed=42,
):
    val_size = max(1, int(len(train_set) * val_ratio))
    train_size = len(train_set) - val_size
    if train_size <= 0:
        raise ValueError("val_ratio is too large.")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_set), generator=generator).tolist()
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_subset = Subset(train_set, train_indices)
    val_subset = Subset(val_set, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader
