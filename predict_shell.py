import os
import argparse
import torch
from model.unet_model import UNet
from utils.dataloader import FoodSegDataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image
import numpy as np
from utils.visualization import save_demo


def parse_args():
    parser = argparse.ArgumentParser(description="Segmentation Prediction")
    parser.add_argument('--dataset', type=str, default='../../data/FoodSeg103/new_dataset', help='Path to dataset')
    parser.add_argument('--model_name', type=str, default='unet', help='Model architecture')
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--num_classes', type=int, default=104)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--pretrain_path', type=str, required=True, help='Path to pretrained weights')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to save predictions')
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

    # Dataloader
    dataset = FoodSegDataset(dataset='val', root_dir=args.dataset,
                              image_transform=image_transform,
                              mask_transform=mask_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

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
    save_dir = os.path.join(args.save_dir, 'predict')
    os.makedirs(save_dir, exist_ok=True)

    # Predict
    with torch.no_grad():
        for idx, (images, masks) in enumerate(loader):
            images, masks = images.to(device), masks.to(device)
            with torch.autocast(device_type=device, enabled=args.use_amp):
                outputs = model(images)

            for i in range(images.shape[0]):
                img = images[i].cpu()
                gt = masks[i].cpu()
                pred = torch.argmax(outputs[i].cpu(), dim=0)
                save_path = os.path.join(save_dir, f"{idx * args.batch_size + i}.png")
                save_demo(img, gt, pred, path=save_path)

    print(f"\nPrediction images saved to: {save_dir}")


if __name__ == '__main__':
    main()