import torch

def save(model,
            optimizer=None,
            train_losses=None,
            val_losses=None,
            train_accuracies=None,
            val_accuracies=None,
            savename="checkpoint",
            index=-1):
    savings = {
        'model_state_dict': model.state_dict()
    }
    if optimizer is not None:
        savings["optimizer_state_dict"] = optimizer.state_dict()
    if train_losses is not None:
        savings['train_losses'] = train_losses
    if val_losses is not None:
        savings['val_losses'] = val_losses
    if train_accuracies is not None:
        savings['train_accuracies'] = train_accuracies
    if val_accuracies is not None:
        savings['val_accuracies'] = val_accuracies
    if index >= 0:
        savename = f"{savename}_{index}"
    torch.save(savings, f"{savename}.pth")
    print(f"Saved in file: {savename}.pth")
