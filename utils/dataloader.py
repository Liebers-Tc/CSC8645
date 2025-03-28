import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
from torch.utils.data import DataLoader


class FoodSegDataset(Dataset):
    def __init__(self, root_dir, dataset, image_transform=None, mask_transform=None):
        super().__init__()
        self.image_dir = Path(root_dir) / dataset / 'images'
        self.mask_dir = Path(root_dir) / dataset / 'masks'
        self.image_list = sorted(list(self.image_dir.glob('*.jpg')))
        self.image_transform = image_transform or T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
        self.mask_transform = mask_transform or T.Compose([
        T.Resize((256, 256), interpolation=Image.NEAREST),
        T.Lambda(lambda m: torch.as_tensor(np.array(m), dtype=torch.long).squeeze())
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


if __name__ == '__main__':
    train_dataset = FoodSegDataset(root_dir='new_dataset', dataset='train')
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataset = FoodSegDataset(root_dir='new_dataset', dataset='val')
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_dataset = FoodSegDataset(root_dir='new_dataset', dataset='test')
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)