import os
import argparse
import torch
from model.unet_model import UNet
from model.skip4_resnetcbamaspp_model import ResNetCBAMASPP as ResNet_skip4
from model.skip3_resnetcbamaspp_model import ResNetCBAMASPP as ResNet_skip3
from model.resnetcbamaspp_model import ResNetCBAMASPP as ResNet
from utils.dataloader import FoodSegDataset
from torch.utils.data import DataLoader
from utils.visualization import Visualizer
from utils.metrics import get_metric
from utils.path_utils import find_lastest_path


def parse_args():
    parser = argparse.ArgumentParser(description="Segmentation Prediction")
    parser.add_argument('--dataset', type=str, default='../../data/FoodSeg103/new_dataset')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--num_classes', type=int, default=104)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pretrain_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--backbone', type=str)

    return parser.parse_args()


def main():
    args = parse_args()

    # Dataloader
    test_loader = DataLoader(FoodSegDataset(root_dir=args.dataset, dataset='test'), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.model_name == 'unet':
        model = UNet(in_channels=args.in_channels, num_classes=args.num_classes).to(device)
    elif args.model_name == 'resnet_skip4':
        model = ResNet_skip4(num_classes=args.num_classes, backbone=args.backbone, encoder_pretrained=False).to(device)
    elif args.model_name == 'resnet_skip3':
        model = ResNet_skip3(num_classes=args.num_classes, backbone=args.backbone, encoder_pretrained=False).to(device)
    elif args.model_name == 'resnet':
        model = ResNet(num_classes=args.num_classes, backbone=args.backbone, encoder_pretrained=True).to(device)
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")

    checkpoint = torch.load(args.pretrain_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Path
    save_dir = args.save_dir or find_lastest_path(root_dir='result')
    os.makedirs(save_dir, exist_ok=True)

    # Visualizer
    visualizer = Visualizer(save_dir=save_dir, save=True, show=False, wandb=args.wandb)
    
    # Metric
    metric_fn = get_metric(['miou', 'dice', 'acc'], num_classes=args.num_classes)
    total_metrics = {}

    # Predict
    with torch.no_grad():
        for step, (images, masks) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device)
            with torch.autocast(device_type=device, enabled=args.use_amp):
                outputs = model(images)

            preds = torch.argmax(outputs, dim=1)
            start_index = step * args.batch_size
            visualizer.save_demo(images, masks, preds, start_index=start_index)

            batch_metrics = metric_fn(outputs, masks)
            for k, v in batch_metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v.item()

    for k, v in total_metrics.items():
        total_metrics[k] = v / len(test_loader)
        print(f"{k}: {total_metrics[k]:.4f}")

    print(f"\nPrediction images saved to: {save_dir}")


if __name__ == '__main__':
    main()