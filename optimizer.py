import torch.optim as optim

def get_optimizer(model, opt):
    """
    Creates an optimizer and learning rate scheduler.

    Args:
        model: The model whose parameters will be optimized.
        opt: A configuration object with attributes:
             - lr: Learning rate
             - weight_decay: Weight decay
             - optimizer: 'adam' or 'sgd'
             - momentum (only for SGD)
             - nesterov (only for SGD)
             - total_epoch: Total training epochs
             - cosine: Boolean, whether to use CosineAnnealingLR

    Returns:
        optimizer: The chosen optimizer (SGD or Adam)
        scheduler: Learning rate scheduler
    """

    base_params = []
    classifier_params = []

    for name, param in model.named_parameters():
        if "classifier" in name:
            classifier_params.append(param)
        else:
            base_params.append(param)

    # Select optimizer
    if opt.optimizer.lower() == "sgd":
        optimizer = optim.SGD([
            {'params': base_params, 'lr': 0.1 * opt.lr},
            {'params': classifier_params, 'lr': opt.lr}
        ], weight_decay=opt.weight_decay, 
           momentum=opt.momentum, 
           nesterov=opt.nesterov)

    elif opt.optimizer.lower() == "adam":
        optimizer = optim.Adam([
            {'params': base_params, 'lr': 0.1 * opt.lr},
            {'params': classifier_params, 'lr': opt.lr}
        ], weight_decay=opt.weight_decay)

    else:
        raise ValueError("Unsupported optimizer! Choose 'adam' or 'sgd'.")

    # Select scheduler
    if opt.cosine:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.total_epoch, eta_min=0.01 * opt.lr)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.total_epoch * 2 // 3, gamma=0.1)

    return optimizer, scheduler
