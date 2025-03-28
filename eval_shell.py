import os
import argparse
import torch
from torch.utils.data import DataLoader
from model.unet_model import UNet
from utils.dataloader import FoodSegDataset
from utils.loss import get_loss
from utils.metrics import get_metric
from trainer import Trainer
from utils.path_utils import get_save_path
import torchvision.transforms as T
from PIL import Image
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Segmentation Evaluation")
    parser.add_argument('--dataset', type=str, default='../../data/FoodSeg103/new_dataset')
    parser.add_argument('--model_name', type=str, default='unet')
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--num_classes', type=int, default=104)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--loss', type=str, default='ce')
    parser.add_argument('--metric', nargs='+', default=['miou', 'dice', 'acc'])
    parser.add_argument('--main_metric', type=str, default='miou')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--pretrain_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--vis_num_sample', type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

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

    # DataLoader
    dataset_args = dict(root_dir=args.dataset, image_transform=image_transform, mask_transform=mask_transform)
    val_loader = DataLoader(FoodSegDataset(dataset='val', **dataset_args), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.model_name == 'unet':
        model = UNet(in_channels=args.in_channels, num_classes=args.num_classes).to(device)
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")

    # Modules
    loss_fn = get_loss(args.loss)
    metric_fn = get_metric(args.metric, num_classes=args.num_classes)
    main_metric = args.main_metric or args.metric[0]

    # Trainer
    trainer = Trainer(model=model,
                      val_loader=val_loader,
                      loss_fn=loss_fn,
                      metric_fn=metric_fn,
                      main_metric=main_metric,
                      device=device,
                      use_amp=args.use_amp,
                      save_dir=args.save_dir,
                      resume_path=args.pretrain_path,
                      wandb=args.wandb,
                      vis_num_sample=args.vis_num_sample)

    # Load weights & Evaluate
    trainer.load_checkpoint()
    val_stats = trainer.run_one_epoch(train=False, epoch=0)

    # Print results
    print("\n[Evaluation Result]")
    for k, v in val_stats.items():
        print(f"{k}: {v:.4f}")


if __name__ == '__main__':
    main()
