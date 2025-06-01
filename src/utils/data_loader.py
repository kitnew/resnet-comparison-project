from torch.utils.data import DataLoader, Dataset, random_split
import os
from PIL import Image
import torch
from torchvision.transforms import v2
import kagglehub
import sys
import logging
from pathlib import Path

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

def create_data_loaders(batch_size=96, num_workers=8):
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
        v2.RandomResizedCrop(224),
        v2.RandomHorizontalFlip(),
        v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
