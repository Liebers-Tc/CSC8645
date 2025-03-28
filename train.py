import torch
from torch.utils.data import DataLoader
from model.unet_model import UNet
from utils.dataloader import FoodSegDataset
from utils.loss import get_loss
from utils.metrics import get_metric
from utils.optim import get_optimizer, get_scheduler
from utils.path_utils import get_save_path
from trainer import Trainer


# Parameters
config = {
    'dataset': '../../data/FoodSeg103/new_dataset',
    'save_dir': None,
    'model_name': 'unet',
    'in_channels': 3,
    'num_classes': 104,
    'epochs': 50,
    'batch_size': 16,
    'num_workers': 12,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'loss': 'ce',
    'metric': ['miou', 'dice', 'acc'],
    'main_metric': 'miou',
    'vis_num_sample': 1,
    'optimizer': 'adamw',
    'scheduler': 'step',
    'early_stopping_patience': 3,
    'use_amp': True,
    'wandb': False,
    'resume_path': None
}

# Dataloader
train_loader = DataLoader(FoodSegDataset(root_dir=config['dataset'], dataset='train'), batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
val_loader = DataLoader(FoodSegDataset(root_dir=config['dataset'], dataset='val'), batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if config['model_name'] == 'unet':
    model = UNet(in_channels=config['in_channels'], num_classes=config['num_classes']).to(device)
else:
    raise ValueError(f"Unsupported model: {config['model_name']}")

# Paths
save_dir = config['save_dir'] or get_save_path(root_dir='result')

# Components
loss_fn = get_loss(config['loss'])

if isinstance(config['metric'], str):
    config['metric'] = [config['metric']]
metric_fn = get_metric(config['metric'], num_classes=config['num_classes'])

main_metric = config.get('main_metric', config['metric'][0])

optimizer = get_optimizer(model, name=config['optimizer'], lr=config['learning_rate'], weight_decay=config['weight_decay'])
scheduler = get_scheduler(optimizer, name=config['scheduler'])

# Trainer
trainer = Trainer(model=model,
                  train_loader=train_loader,
                  val_loader=val_loader,
                  loss_fn=loss_fn,
                  metric_fn=metric_fn,
                  main_metric=main_metric,
                  optimizer=optimizer,
                  scheduler=scheduler,
                  device=device,
                  use_amp=config['use_amp'],
                  early_stopping_patience=config['early_stopping_patience'],
                  save_dir=save_dir,
                  resume_path=config['resume_path'],
                  wandb=config['wandb'],
                  vis_num_sample=config['vis_num_sample'])

# Load weight
if config['resume_path']:
    trainer.load_checkpoint()

# Train
trainer.fit(config['epochs'])
