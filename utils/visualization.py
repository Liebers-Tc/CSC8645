import os
import torch
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, save_dir, save=True, show=False, wandb=False):
        self.save_dir = save_dir
        self.save = save
        self.show = show
        self.wandb = wandb
        os.makedirs(save_dir, exist_ok=True)
        
    @staticmethod
    def denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        mean = torch.tensor(mean).view(3, 1, 1).to(image.device)
        std = torch.tensor(std).view(3, 1, 1).to(image.device)
        return image * std + mean

    def plot_demo(self, image, gt_mask, pred_mask, save_path=None):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        image = self.denormalize(image)
        axes[0].imshow(image.permute(1, 2, 0).cpu())
        axes[0].set_title("Image")
        axes[1].imshow(gt_mask.cpu(), cmap='tab20')
        axes[1].set_title("Ground Truth")
        axes[2].imshow(pred_mask.cpu(), cmap='tab20')
        axes[2].set_title("Prediction")
        for ax in axes: ax.axis("off")

        if self.save:
            fig.savefig(save_path)
            if self.wandb:
                import wandb
                wandb.log({f"Prediction/{os.path.basename(save_path)}": wandb.Image(save_path)})

        if self.show:
            plt.show()

        plt.close(fig)

    def save_demo(self, images, gts, preds, start_index=0):
        for i in range(len(images)):
            save_path = os.path.join(self.save_dir, f"sample_{start_index + i + 1}.png")
            self.plot_demo(images[i], gts[i], preds[i], save_path=save_path)
