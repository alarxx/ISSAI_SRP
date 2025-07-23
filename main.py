import os
import math

import numpy as np

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import tkinter
# tkinter._test()
# print(matplotlib.get_backend())

import torch
from torch import nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import torch.nn.functional as F
import torch.optim as optim

# utils
from utils.data.CustomImageDataset import CustomImageDataset
from utils.devices import print_available_devices
print_available_devices()
from utils.plot import plot4
from utils.save import save
from utils.trainer import Trainer
from utils.model.MLP import MLP

# ---

if __name__ == "__main__":

    CHANNELS, HEIGHT, WIDTH = 3, 28, 28

    transform = transforms.Compose([
        transforms.Resize((HEIGHT, WIDTH)),  # например
        transforms.ToTensor(), # tensor(C, H, W) in RGB
    ])

    dataset = CustomImageDataset(root_dir="dataset/CMNIST", transform=transform)

    # Как разделить на Train, Val, Test
    # validation_data = K-Fold Cross Validation on training_data
    training_data, test_data = dataset.train_test_split(test_size=0.25, random_state=42)

    # ---

    BATCH_SIZE = 256

    # Create data loaders.
    # Data Loader wraps an iterable over dataset
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    for X, y in test_dataloader:
    # X, y = next(iter(test_dataloader))
        print(f"Shape of X [N, C, H, W]: {X.shape}") # torch.Size([64, 1, 28, 28]) torch.float32 [0; 1]
        print(f"Shape of y: {y.shape} {y.dtype}") # torch.Size([64]) torch.int64
        break

    # ---

    def visualize(dataset, classes):
        figure = plt.figure(figsize=(8, 8))
        cols, rows = 3, 3
        for i in range(1, cols * rows + 1):
            sample_idx = torch.randint(len(dataset), size=(1,)).item()
            img, label = dataset[sample_idx]
            # print(img.shape)
            figure.add_subplot(rows, cols, i)
            plt.title(classes[label])
            plt.axis("off")
            plt.imshow(img.permute(1, 2, 0)) # tensor(C, H, W), а метод принимает img(H, W, C)
        plt.show()

    visualize(dataset, dataset.classes)

    # ---

    # device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # ---

    model = MLP(inp=CHANNELS*HEIGHT*WIDTH, out=len(dataset.classes)).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3) # *math.sqrt(BATCH_SIZE) # doesn't work with AdamW
    trainer = Trainer(model, optimizer, device)

    EPOCHS = 5
    trainer.fit(
        train_dataloader=train_dataloader,
        val_dataloader=test_dataloader, # actually has to be val_dataloader, not test_dataloader
        epochs=EPOCHS,
        savename="checkpoint"
    )

    # --- Load ---

    model = MLP(inp=CHANNELS*HEIGHT*WIDTH, out=len(dataset.classes)).to(device)

    checkpoint = torch.load(f"checkpoint_{EPOCHS-1}.pth", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    plot4(
        trloss=checkpoint["train_losses"],
        tracc=checkpoint["train_accuracies"],
        tloss=checkpoint["val_losses"],
        tacc=checkpoint["val_accuracies"]
    )

    print(f"Test loaded model\n-------------------------------")
    Trainer.test(model, test_dataloader, device)


