import argparse
import torch
from torch.utils.data import DataLoader
from model.unet_model import UNet
from utils.dataloader import FoodSegDataset
from utils.loss import get_loss
from utils.metrics import get_metric
from utils.optim import get_optimizer, get_scheduler
from trainer import Trainer
from utils.path_utils import get_save_path


def parse_args():
    parser = argparse.ArgumentParser(description="Segmentation Training")
    parser.add_argument('--dataset', type=str, default='../../data/FoodSeg103/new_dataset')
    parser.add_argument('--model_name', type=str, default='unet') 
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--num_classes', type=int, default=104)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=12)

    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--scheduler', type=str, default='step')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--loss', type=str, default='ce')
    parser.add_argument('--metric', nargs='+', default=['miou', 'dice', 'acc'])
    parser.add_argument('--main_metric', type=str, default='miou')
    
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--pretrain_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--vis_num_sample', type=int, default=1)
    parser.add_argument('--early_stopping_patience', type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()

    # Auto create save_dir
    save_dir = args.save_dir or get_save_path(root_dir='result')

    # Dataloader
    train_loader = DataLoader(FoodSegDataset(root_dir=args.dataset, dataset='train'), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(FoodSegDataset(root_dir=args.dataset, dataset='val'), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.model_name == 'unet':
        model = UNet(in_channels=args.in_channels, num_classes=args.num_classes).to(device)
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")

    # Loss & Metrics
    loss_fn = get_loss(args.loss)
    metric_fn = get_metric(args.metric, num_classes=args.num_classes)
    main_metric = args.main_metric or args.metric[0]

    # Optimizer & Scheduler
    optimizer = get_optimizer(model, args.optimizer, args.learning_rate, args.weight_decay)
    scheduler = get_scheduler(optimizer, args.scheduler)

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
                      use_amp=args.use_amp,
                      save_dir=save_dir,
                      resume_path=args.pretrain_path,
                      early_stopping_patience=args.early_stopping_patience,
                      wandb=args.wandb,
                      vis_num_sample=args.vis_num_sample)

    # Start Training
    trainer.load_checkpoint()
    trainer.fit(args.epochs)


if __name__ == '__main__':
    main()