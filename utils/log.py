import os
import matplotlib.pyplot as plt

class Logger:
    def __init__(self, save_dir, save=True, show=True, wandb=False, wandb_project=None, wandb_run_name=None):
        self.save_dir = save_dir
        self.save = save
        self.show = show
        self.wandb = wandb
        os.makedirs(save_dir, exist_ok=True)
        self.log_path = os.path.join(save_dir, 'log.txt')

        with open(self.log_path, 'w') as f:
            f.write("epoch,train_loss,train_metric,val_loss,val_metric\n")

        if self.wandb:
            import wandb
            wandb.init(project=wandb_project, name=wandb_run_name)

    def log_scalar(self, tag, value, step):
        if self.wandb:
            import wandb
            wandb.log({tag: value}, step=step)

    def log_text(self, epoch, train_loss, train_metric, val_loss=None, val_metric=None):
        with open(self.log_path, 'a') as f:
            line = f"{epoch},{train_loss:.4f},{train_metric:.4f}"
            if val_loss is not None:
                line += f",{val_loss:.4f},{val_metric:.4f}"
            f.write(line + "\n")

    def plot_curve(self, x, y1, y2, title, ylabel, labels, save_path):
        plt.figure()
        plt.plot(x, y1, label=labels[0])
        plt.plot(x, y2, label=labels[1])
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.legend()
        if self.save:
            plt.savefig(save_path)
            if self.wandb:
                import wandb
                wandb.log({title: wandb.Image(save_path)})
        if self.show:
            plt.show()
        plt.close()

    def save_curve(self, train_losses, val_losses, train_metrics, val_metrics, metric_name='mIoU'):
        epochs = range(1, len(train_losses) + 1)
        self.plot_curve(epochs, train_losses, val_losses, 'Loss Curve', 'Loss',
                        ['Train Loss', 'Val Loss'], os.path.join(self.save_dir, 'loss_curve.png'))
        self.plot_curve(epochs, train_metrics, val_metrics, f'{metric_name} Curve', metric_name,
                        [f'Train {metric_name}', f'Val {metric_name}'],
                        os.path.join(self.save_dir, f'{metric_name.lower()}_curve.png'))
