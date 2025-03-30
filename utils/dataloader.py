import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np


def mask_to_tensor(m):
    return torch.as_tensor(np.array(m), dtype=torch.long).squeeze()

class FoodSegDataset(Dataset):
    def __init__(self, root_dir, dataset):
        super().__init__()
        self.image_dir = Path(root_dir) / dataset / 'images'
        self.mask_dir = Path(root_dir) / dataset / 'masks'
        self.image_list = sorted(list(self.image_dir.glob('*.jpg')))
        
        self.image_transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
                        ])
        self.mask_transform = T.Compose([
            T.Resize((256, 256), interpolation=Image.NEAREST),
            T.Lambda(mask_to_tensor)
            ])
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        mask_path = self.mask_dir / image_path.with_suffix('.png').name

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path)

        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        return image, mask