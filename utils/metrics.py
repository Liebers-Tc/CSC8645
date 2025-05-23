import torch


class MeanIoU:
    def __init__(self, num_classes, smooth=1e-6, ignore_index=None):
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def __call__(self, preds, targets):
        preds = torch.argmax(preds, dim=1)  # [B, H, W]
        targets = targets.to(preds.device)
        ious = []

        for cls in range(self.num_classes):
            if cls == self.ignore_index:
                continue
            pred_cls = (preds == cls).float()
            target_cls = (targets == cls).float()
            intersection = (pred_cls * target_cls).sum()
            union = pred_cls.sum() + target_cls.sum() - intersection
            if union == 0:
                ious.append(torch.tensor(float('nan'), device=preds.device))  # class not exist, default nan // Avoid empty class interference average
            else:
                ious.append((intersection + self.smooth) / (union + self.smooth))

        return torch.nanmean(torch.stack(ious))  # return mIoU


class DiceScore:
    def __init__(self, num_classes, smooth=1e-6):
        self.num_classes = num_classes
        self.smooth = smooth

    def __call__(self, preds, targets):
        """
        preds: [B, C, H, W] logits
        targets: [B, H, W] (long)
        """
        preds = torch.argmax(preds, dim=1)  # [B, H, W]
        targets = targets.to(preds.device)
        dice_scores = []

        for cls in range(self.num_classes):
            pred_cls = (preds == cls).float()
            target_cls = (targets == cls).float()
            intersection = (pred_cls * target_cls).sum()
            union = pred_cls.sum() + target_cls.sum()
            if union == 0:
                dice_scores.append(torch.tensor(float('nan'), device=preds.device))  # class not exist, default nan
            else:
                dice_scores.append((2 * intersection + self.smooth) / (union + self.smooth))
        
        return torch.nanmean(torch.stack(dice_scores))
    

class PixelAccuracy:
    def __call__(self, preds, targets):
        """
        preds: [B, C, H, W] (logits)
        targets: [B, H, W] (long)
        """
        preds = torch.argmax(preds, dim=1)  # [B, H, W]
        correct = (preds == targets).float()
        
        return correct.sum() / correct.numel()


class MetricCollection:
    def __init__(self, metrics: dict):
        self.metrics = metrics

    def __call__(self, preds, targets):
        return {name: metric(preds, targets) for name, metric in self.metrics.items()}

    def keys(self):
        return list(self.metrics.keys())


def get_metric(names=['miou', 'dice', 'acc'], num_classes=104, **kwargs):
    if isinstance(names, str):
        names = [names]

    available = {
        'miou': MeanIoU(num_classes=num_classes, **kwargs),
        'dice': DiceScore(num_classes=num_classes, **kwargs),
        'acc': PixelAccuracy()
    }

    selected = {}
    for name in names:
        name = name.lower()
        if name not in available:
            raise ValueError(f"Unsupported metric: {name}")
        selected[name] = available[name]

    return MetricCollection(selected)