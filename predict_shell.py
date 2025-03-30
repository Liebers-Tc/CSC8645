import os
import argparse
import torch
from model.unet_model import UNet
from utils.dataloader import FoodSegDataset
from torch.utils.data import DataLoader
from utils.visualization import Visualizer
from utils.path_utils import find_lastest_path


def parse_args():
    parser = argparse.ArgumentParser(description="Segmentation Prediction")
    parser.add_argument('--dataset', type=str, default='../../data/FoodSeg103/new_dataset')
    parser.add_argument('--model_name', type=str, default='unet')
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--num_classes', type=int, default=104)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pretrain_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default=None, required=True)
    parser.add_argument('--vis_num_sample', type=int, default=1)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--wandb', action='store_true')

    return parser.parse_args()


def main():
    args = parse_args()

    # Dataloader
    test_loader = DataLoader(FoodSegDataset(root_dir=args.dataset, dataset='test'), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.model_name == 'unet':
        model = UNet(in_channels=args.in_channels, num_classes=args.num_classes).to(device)
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")

    checkpoint = torch.load(args.pretrain_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Path
    save_dir = args.save_dir or find_lastest_path(root_dir='result')
    os.makedirs(save_dir, exist_ok=True)

    # Visualizer
    visualizer = Visualizer(save_dir=save_dir, save=True, show=False, wandb=args.wandb, num_sample=args.vis_num_sample)

    # Predict
    with torch.no_grad():
        for idx, (images, masks) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device)
            with torch.autocast(device_type=device, enabled=args.use_amp):
                outputs = model(images)

            outputs = torch.argmax(outputs, dim=1)
            visualizer.save_demo(images, masks, outputs)

    print(f"\nPrediction images saved to: {save_dir}")


if __name__ == '__main__':
    main()