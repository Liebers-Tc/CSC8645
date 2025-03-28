import os
import torch
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from utils.log import Logger
from utils.visualization import Visualizer


class Trainer:
    def __init__(self, model, train_loader=None, val_loader=None, device=None,
                 loss_fn=None, metric_fn=None, main_metric=None,
                 optimizer=None, scheduler=None, use_amp=True, 
                 save_dir=None, resume_path=None,
                 early_stopping_patience=None,
                 wandb=False, vis_num_sample=1):

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.main_metric = main_metric
        self.greater_is_better = main_metric != 'loss'

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.use_amp = use_amp
        self.scaler = GradScaler(enabled=use_amp)

        # subfolder: checkpoint, log, visualization
        self.save_dir = save_dir
        self.resume_path = resume_path
        self.ckpt_dir = os.path.join(save_dir, 'checkpoint')
        self.log_dir = os.path.join(save_dir, 'log')
        self.vis_dir = os.path.join(save_dir, 'visualization')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)

        self.best_val_metric = -float('inf') if self.greater_is_better else float('inf')
        self.early_stopping_patience = early_stopping_patience
        self.early_stop_counter = 0

        self.logger = Logger(save_dir=self.log_dir, wandb=wandb)
        self.visualizer = Visualizer(save_dir=self.vis_dir, wandb=wandb, num_sample=vis_num_sample)

    def run_one_epoch(self, train=True, epoch=0):
        mode = 'Train' if train else 'Val'
        loader = self.train_loader if train else self.val_loader
        self.model.train() if train else self.model.eval()

        total_loss = 0.0
        total_metrics = {}
        loop = tqdm(loader, desc=mode, leave=False)

        with torch.set_grad_enabled(train):
            for i, (images, masks) in enumerate(loop):
                images, masks = images.to(self.device), masks.to(self.device)

                with autocast(enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, masks)
                    metrics = self.metric_fn(outputs, masks)

                if train:
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                total_loss += loss.item()

                for k, v in metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0.0) + float(v)

                loop.set_postfix({"loss": loss.item(), **{k: float(v) for k, v in metrics.items()}})

                if not train and i == 0:
                    self.visualizer.plot_demo(images[0], masks[0], torch.argmax(outputs[0], dim=0), step=epoch)

        if train and self.scheduler:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(total_loss)
            else:
                self.scheduler.step()

        avg_metrics = {k: v / len(loader) for k, v in total_metrics.items()}
        avg_metrics['loss'] = total_loss / len(loader)
        return avg_metrics

    def save_checkpoint(self, filename="checkpoint.pt", epoch=None):
        os.makedirs(self.ckpt_dir, exist_ok=True)
        save_path = os.path.join(self.ckpt_dir, filename)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'epoch': epoch,
            'best_val_metric': self.best_val_metric
        }

        if self.optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, save_path)
        print(f"\nSuccessfully saved model to : {save_path}")

    def load_checkpoint(self):
        if not self.resume_path or not os.path.exists(self.resume_path):
            print(f"\nNot found model from: {self.resume_path}")
            self.start_epoch = 0
            return

        checkpoint = torch.load(self.resume_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.start_epoch = checkpoint.get('epoch', 0)
        self.best_val_metric = checkpoint.get('best_val_metric', self.best_val_metric)

        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"\nSuccessfully load model from: {self.resume_path}")

    def fit(self, epochs):
        start = getattr(self, 'start_epoch', 0)
        for epoch in range(start, epochs):
            print(f"\n[Epoch {epoch + 1}/{epochs}]")

            train_stats = self.run_one_epoch(train=True)
            val_stats = self.run_one_epoch(train=False, epoch=epoch)

            self.logger.log_scalar("Loss/train", train_stats['loss'], epoch)
            self.logger.log_scalar("Loss/val", val_stats['loss'], epoch)
            for k in val_stats:
                self.logger.log_scalar(f"Metric/val/{k}", val_stats[k], epoch)
            for k in train_stats:
                self.logger.log_scalar(f"Metric/train/{k}", train_stats[k], epoch)

            self.logger.log_text(epoch, train_stats['loss'], train_stats, val_stats['loss'], val_stats)

            main_val = val_stats.get(self.main_metric)
            if main_val is None:
                print(f"\n[Warning] Main metric '{self.main_metric}' not found in validation metrics list {self.metric_fn.keys()}")
                continue

            is_better = (main_val > self.best_val_metric) if self.greater_is_better else (main_val < self.best_val_metric)

            if is_better:
                self.best_val_metric = main_val
                self.early_stop_counter = 0
                self.save_checkpoint(filename="best.pt", epoch=epoch + 1)
            else:
                self.early_stop_counter += 1
                if self.early_stopping_patience and self.early_stop_counter >= self.early_stopping_patience:
                    print("\nEarly stopping triggered.")
                    break

        self.save_checkpoint(filename="last.pt", epoch=epoch + 1)
