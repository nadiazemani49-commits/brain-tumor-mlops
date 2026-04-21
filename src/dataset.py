import random
from pathlib import Path
from typing import Literal
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

CLASSES   = ['glioma', 'meningioma', 'pituitary', 'notumor']
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
SEED = 42

def get_transforms(mode: str):
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

class BrainTumorDataset(Dataset):
    def __init__(self, root, split, mode='train', sample_fraction=1.0):
        self.transform = get_transforms(mode)
        self.samples   = []
        for cls in CLASSES:
            imgs = sorted((Path(root) / split / cls).glob('*.jpg'))
            if sample_fraction < 1.0:
                random.seed(SEED)
                imgs = random.sample(imgs, max(4, int(len(imgs) * sample_fraction)))
            for p in imgs:
                self.samples.append((p, CLASS2IDX[cls]))
        print(f'[{split}] {len(self.samples)} images ({sample_fraction*100:.0f}%)')

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        p, label = self.samples[idx]
        return self.transform(Image.open(p).convert('RGB')), label

def get_dataloaders(data_root, batch_size=32, num_workers=2, sample_fraction=0.1):
    train_ds = BrainTumorDataset(data_root, 'Training', 'train', sample_fraction)
    test_ds  = BrainTumorDataset(data_root, 'Testing',  'test',  sample_fraction)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
