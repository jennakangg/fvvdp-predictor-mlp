import wandb

def log_gradients(model, epoch, batch_idx, train_loader):
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            gradients[f'grads/{name}'] = wandb.Histogram(parameter.grad.cpu().numpy())
        else:
            gradients[f'grads/{name}'] = wandb.Histogram([])  # Log empty if no gradient

    wandb.log(gradients, step=epoch * len(train_loader) + batch_idx)