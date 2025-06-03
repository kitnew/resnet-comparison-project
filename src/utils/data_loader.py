from torch.utils.data import DataLoader, Dataset, random_split, RandomSampler
import os
from PIL import Image
import torch
from torchvision.transforms import v2
import kagglehub
import sys
import logging
from pathlib import Path
import numpy as np

class ImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if os.path.isfile(os.path.join(class_dir, img_name)):
                    self.samples.append((os.path.join(class_dir, img_name), self.class_to_idx[class_name]))
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

class ClassBalancedLoss:
    """
    Class-balanced loss function for handling imbalanced datasets.
    Implementation based on the paper "Class-Balanced Loss Based on Effective Number of Samples"
    https://arxiv.org/abs/1901.05555
    """
    def __init__(self, class_counts, beta=0.9999, loss_type='cross_entropy'):
        """
        Initialize the class-balanced loss.
        
        Args:
            class_counts: List containing count of samples for each class
            beta: Hyperparameter for class-balanced loss (default: 0.9999)
            loss_type: Base loss function type (default: 'cross_entropy')
        """
        self.class_counts = class_counts
        self.beta = beta
        self.loss_type = loss_type
        
        # Calculate effective number of samples
        effective_num = 1.0 - np.power(self.beta, class_counts)
        
        # Calculate weights for each class
        self.class_weights = (1.0 - self.beta) / np.array(effective_num)
        self.class_weights = self.class_weights / np.sum(self.class_weights) * len(class_counts)
        self.class_weights = torch.tensor(self.class_weights, dtype=torch.float32)
    
    def get_class_weights(self):
        """Return the class weights tensor."""
        return self.class_weights

def create_data_loaders(batch_size=96, num_workers=8, use_class_balancing=True):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Get dataset path from kagglehub
    logger.info("Downloading dataset from kagglehub...")
    dataset_path = kagglehub.dataset_download("dimensi0n/imagenet-256")
    logger.info(f"Dataset downloaded to: {dataset_path}")
    
    # Check if path exists and contains files
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")
    
    # Define transformations for ImageNet
    train_transform = v2.Compose([
        v2.RandomResizedCrop(224, scale=(0.08, 1.0)),
        v2.RandomHorizontalFlip(),
        v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        v2.MixUp(alpha=0.2),
        v2.CutMix(alpha=1.0),
        v2.AutoAugment(policy=v2.AutoAugmentPolicy.IMAGENET),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        v2.RandomErasing(p=0.25, scale=(0.02, 0.2))
    ])
    
    val_test_transform = v2.Compose([
        v2.Resize(224),
        v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create full dataset with the valid directory
    full_dataset = ImageNetDataset(dataset_path, transform=None)
    logger.info(f"Created dataset with {len(full_dataset)} samples")
    
    if len(full_dataset) == 0:
        raise ValueError("Dataset is empty - no images found")
        
    # Split dataset: 60% train, 20% val, 20% test
    total_size = len(full_dataset)
    train_size = int(0.6 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(688243)
    )
    
    # Apply transformations
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_test_transform
    test_dataset.dataset.transform = val_test_transform
    
    # Create data loaders
    logger.info(f"Creating data loaders with batch size {batch_size}")
    
    # Calculate class counts for class-balanced loss
    class_counts = [0] * len(full_dataset.class_to_idx)
    for _, label in full_dataset.samples:
        class_counts[label] += 1
    
    class_balancer = None
    if use_class_balancing:
        # Instead of WeightedRandomSampler, use RandomSampler and class-balanced loss
        logger.info("Using RandomSampler with class-balanced loss for training data")
        class_balancer = ClassBalancedLoss(class_counts)
        
        # Use RandomSampler for the training loader
        train_sampler = RandomSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=train_sampler,
            num_workers=num_workers
        )
    else:
        # Use regular random sampling
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers
        )
    
    # Validation and test loaders (no need for class balancing here)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, class_balancer
