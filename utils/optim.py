
from torch import optim


def get_optimizer(model, name='adam', lr=1e-3, weight_decay=1e-4, **kwargs):
    name = name.lower()
    params = model.parameters()

    if name == 'adam':
        return optim.Adam(params, lr=lr, weight_decay=weight_decay, **kwargs)
    elif name == 'adamw':
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay, **kwargs)
    elif name == 'sgd':
        return optim.SGD(params, lr=lr, weight_decay=weight_decay,
                         momentum=0.9, nesterov=True, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")


def get_scheduler(optimizer, name='step', **kwargs):
    name = name.lower()

    if name == 'step':
        # default: step_size=10, gamma=0.1
        return optim.lr_scheduler.StepLR(optimizer, step_size=kwargs.get('step_size', 10), gamma=kwargs.get('gamma', 0.1))
    elif name == 'cosine':
        # default: T_max=50
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=kwargs.get('T_max', 50))
    elif name == 'plateau':
        # default: mode='min', patience=5
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=kwargs.get('mode', 'min'),
                                                    patience=kwargs.get('patience', 5),
                                                    factor=kwargs.get('factor', 0.1))
    elif name == 'none':
        return None
    else:
        raise ValueError(f"Unsupported scheduler: {name}")