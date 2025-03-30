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

        if self.wandb:
            import wandb
            wandb.init(project=wandb_project, name=wandb_run_name)

    def log_scalar(self, tag, value, step):
        if self.wandb:
            import wandb
            wandb.log({tag: value}, step=step)

    def log_text(self, line):
        with open(self.log_path, 'a') as f:
            f.write(line + '\n')

    def plot_curve(self, x, y1, y2=None, title='', ylabel='', labels=None, save_path='curve.png'):
        plt.figure()
        plt.plot(x, y1, label=labels[0] if labels else 'Line 1')
        if y2:
            plt.plot(x, y2, label=labels[1] if labels and len(labels) > 1 else 'Line 2')
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

    def save_curve(self, train_losses, val_losses, val_metrics_dict=None):
        epochs = range(1, len(train_losses) + 1)
        self.plot_curve(epochs, train_losses, val_losses, 'Loss Curve', 'Loss',
                        ['Train Loss', 'Val Loss'], os.path.join(self.save_dir, 'loss_curve.png'))
        if val_metrics_dict:
            for metric_name, val_values in val_metrics_dict.items():
                self.plot_curve(epochs, val_values, None,
                                f"Val {metric_name} Curve", metric_name.upper(),
                                [f"Val {metric_name}"],
                                os.path.join(self.save_dir, f"val_{metric_name.lower()}_curve.png"))
