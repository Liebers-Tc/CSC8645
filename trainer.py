import os
import torch
from torch.amp import autocast, GradScaler
from utils.log import Logger
from utils.visualization import Visualizer


class Trainer:
    def __init__(self, model, train_loader, val_loader, 
                 loss_fn, metric_fn, main_metric, 
                 optimizer, scheduler, 
                 early_stopping_patience, 
                 device, use_amp, 
                 save_dir, resume_path=None,  
                 vis_num_sample=1, wandb=False):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.main_metric = main_metric
        self.greater_is_better = main_metric != 'loss'

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping_patience = early_stopping_patience
        self.early_stop_counter = 0

        self.device = device
        self.use_amp = use_amp
        self.save_dir = save_dir
        self.resume_path = resume_path
        
        self.vis_num_sample = vis_num_sample
        self.wandb = wandb

        self.scaler = GradScaler(enabled=use_amp)
        self.best_score = -float('inf') if self.greater_is_better else float('inf')

        self.ckpt_dir = os.path.join(save_dir, 'checkpoint')
        self.log_dir = os.path.join(save_dir, 'log')
        self.vis_dir = os.path.join(save_dir, 'visualization')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)
        self.logger = Logger(save_dir=self.log_dir, wandb=self.wandb)
        self.visualizer = Visualizer(save_dir=self.vis_dir, save=True, show=False, wandb=wandb, num_sample=self.vis_num_sample)

        self.train_history = {"loss": []}
        self.val_history = {"loss": []}
        # self.train_metrics = {}
        self.val_metrics = {}

    def run_one_epoch(self, train=True):
        dataloader = self.train_loader if train else self.val_loader
        self.model.train() if train else self.model.eval()

        total_loss = 0.0
        total_metric = {}

        with torch.set_grad_enabled(train):
            for i, (images, masks) in enumerate(dataloader):
                images, masks = images.to(self.device), masks.to(self.device)

                with autocast(device_type=self.device, enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, masks)
                    metrics = self.metric_fn(outputs, masks) if not train else {}

                if train:
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                total_loss += loss.item()

                for k, v in metrics.items():
                    total_metric[k] = total_metric.get(k, 0.0) + v.item()

                if not train and i == 0:
                    outputs = torch.argmax(outputs, dim=1)
                    self.visualizer.save_demo(images, masks, outputs)

        avg_loss = total_loss / len(dataloader)
        avg_metrics = {k: v / len(dataloader) for k, v in total_metric.items()}
        return {"loss": avg_loss, **avg_metrics}
    
    def save_checkpoint(self, epoch, filename):
        filepath = os.path.join(self.ckpt_dir, f"model_{filename}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
            # "train_history": self.train_history,
            # "val_history": self.val_history,
            # "best_score": self.best_score
            }, filepath)
        print(f"Model saved to {filepath}")

    def load_checkpoint(self):
        if self.resume_path is None:
            self.start_epoch = 0
            return
        elif self.resume_path and not os.path.exists(self.resume_path):
            raise FileNotFoundError(f"Checkpoint or Pre-trained weight not found at: {self.resume_path}")
        else:
            checkpoint = torch.load(self.resume_path, map_location=self.device)
            self.start_epoch = checkpoint["epoch"]
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            # self.train_history = checkpoint.get("train_history", {"loss": []})
            # self.val_history = checkpoint.get("val_history", {"loss": []})
            # self.best_score = checkpoint.get("best_score", self.best_score)

            print(f"Resume model from: {self.resume_path}")

    def fit(self, epochs):
        for epoch in range(self.start_epoch + 1, epochs + 1):
            print(f"\n[Epoch {epoch}/{epochs}]")

            train_stats = self.run_one_epoch(train=True)
            val_stats = self.run_one_epoch(train=False)

            train_loss = train_stats["loss"]
            val_loss = val_stats["loss"]

            self.train_history["loss"].append(train_loss)
            self.val_history["loss"].append(val_loss)

            self.logger.log_scalar("Loss/train", train_loss, step=epoch)
            self.logger.log_scalar("Loss/val", val_loss, step=epoch)

            for k, v in val_stats.items():
                if k != "loss":
                    self.val_metrics.setdefault(k, []).append(v) # append val metrics to self.val_metrics
                    self.logger.log_scalar(f"{k}/val", v, step=epoch)

            log_msg = f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, " + \
                      ", ".join([f"{k}={v:.4f}" for k, v in val_stats.items() if k != 'loss'])
            print(log_msg)
            self.logger.log_text(log_msg)

            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_stats.get(self.main_metric, val_stats['loss']))
                else:
                    self.scheduler.step()
            
            main_value = val_stats[self.main_metric]
            value = (main_value > self.best_score) if self.greater_is_better else (main_value < self.best_score)
            if value:
                self.best_score = main_value
                self.early_stop_counter = 0
                self.save_checkpoint(epoch, filename="best")
            else:
                self.early_stop_counter += 1
                if self.early_stopping_patience and self.early_stop_counter >= self.early_stopping_patience:
                    print("Early stopping triggered.")
                    break

        self.save_checkpoint(epoch, filename="final")
        self.logger.save_curve(self.train_history["loss"], self.val_history["loss"], val_metrics_dict=self.val_metrics)




