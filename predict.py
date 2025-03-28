import os
import torch
from model.unet_model import UNet
from utils.dataloader import FoodSegDataset
from torch.utils.data import DataLoader
from utils.visualization import Visualizer

# Parameters
config = {
    'dataset': '../../data/FoodSeg103/new_dataset',
    'model_name': 'unet',
    'in_channels': 3,
    'num_classes': 104,
    'batch_size': 8,
    'num_workers': 8,
    'use_amp': True,
    'pretrain_path': 'result/run_1/checkpoint/best.pt',
    'save_dir': 'result/run_1'
}

# Dataloader
dataset = FoodSegDataset(root_dir=config['dataset'], dataset='test')
loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if config['model_name'] == 'unet':
    model = UNet(in_channels=config['in_channels'], num_classes=config['num_classes']).to(device)
else:
    raise ValueError(f"Unsupported model: {config['model_name']}")

checkpoint = torch.load(config['pretrain_path'], map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Path
save_dir = os.path.join(config['save_dir'], 'predict')
os.makedirs(save_dir, exist_ok=True)

# Visualizer
visualizer = Visualizer(save_dir=save_dir)

# Predict
with torch.no_grad():
    for idx, (images, masks) in enumerate(loader):
        images, masks = images.to(device), masks.to(device)
        with torch.autocast(device_type=device, enabled=config['use_amp']):
            outputs = model(images)

        for i in range(images.shape[0]):
            img = images[i].cpu()
            gt = masks[i].cpu()
            pred = torch.argmax(outputs[i].cpu(), dim=0)
            visualizer.save_demo(img, gt, pred, path=os.path.join(save_dir, f"{idx * config['batch_size'] + i}.png"))

print(f"\nPrediction images saved to: {save_dir}")