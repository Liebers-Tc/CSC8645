import os
import torch
from torch.utils.data import DataLoader
from model.unet_model import UNet
from utils.dataloader import FoodSegDataset
from utils.loss import get_loss
from utils.metrics import get_metric
from trainer import Trainer
import torchvision.transforms as T
from PIL import Image
import numpy as np

# Parameters
config = {
    'dataset': '../../data/FoodSeg103/new_dataset',
    'model_name': 'unet',
    'in_channels': 3,
    'num_classes': 104,
    'batch_size': 16,
    'num_workers': 12,
    'loss': 'ce',
    'metric': ['miou', 'dice', 'acc'],
    'main_metric': 'miou',
    'vis_num_sample': 1,
    'use_amp': True,
    'wandb': False,
    'pretrain_path': 'result/run_1/checkpoint/best.pt',
    'save_dir': 'result/run_1'
}

# Transforms
image_transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
mask_transform = T.Compose([
    T.Resize((256, 256), interpolation=Image.NEAREST),
    T.Lambda(lambda m: torch.as_tensor(np.array(m), dtype=torch.long).squeeze())
])

# Dataloader
dataset_args = dict(root_dir=config['dataset'], image_transform=image_transform, mask_transform=mask_transform)
val_loader = DataLoader(FoodSegDataset(dataset='val', **dataset_args), batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if config['model_name'] == 'unet':
    model = UNet(in_channels=config['in_channels'], num_classes=config['num_classes']).to(device)
else:
    raise ValueError(f"Unsupported model: {config['model_name']}")

# Components
loss_fn = get_loss(config['loss'])
metric_fn = get_metric(config['metric'], num_classes=config['num_classes'])
main_metric = config.get('main_metric', config['metric'][0])

# Trainer
trainer = Trainer(model=model,
                  val_loader=val_loader,
                  loss_fn=loss_fn,
                  metric_fn=metric_fn,
                  main_metric=main_metric,
                  device=device,
                  use_amp=config['use_amp'],
                  save_dir=config['save_dir'],
                  resume_path=config['pretrain_path'],
                  wandb=config['wandb'],
                  vis_num_sample=config['vis_num_sample'])

# Load weight
trainer.load_checkpoint()

# Evaluation
val_stats = trainer.run_one_epoch(train=False, epoch=0)
print("\n[Evaluation Result]")
for k, v in val_stats.items():
    print(f"{k}: {v:.4f}")
