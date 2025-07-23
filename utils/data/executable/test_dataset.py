import os

import numpy as np

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import tkinter
# tkinter._test()
# print(matplotlib.get_backend())

import torch
# import torch.nn as nn
from torch import nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import torch.nn.functional as F
import torch.optim as optim

from utils.CustomImageDataset import CustomImageDataset

# ---

if __name__ == "__main__":

    batch_size = 4

    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # например
        transforms.ToTensor(),
    ])

    dataset = CustomImageDataset(root_dir="CMNIST", transform=transform)

    # ---

    def visualize(dataset):
        figure = plt.figure(figsize=(8, 8))
        cols, rows = 3, 3
        for i in range(1, cols * rows + 1):
            sample_idx = torch.randint(len(dataset), size=(1,)).item()
            img, label = dataset[sample_idx]
            print(img.shape)
            figure.add_subplot(rows, cols, i)
            plt.title(dataset.classes[label])
            plt.axis("off")
            plt.imshow(img.permute(1, 2, 0)) # tensor(C, H, W), а метод принимает img(H, W, C)
        plt.show()

    visualize(dataset)

    # ---

    # Create data loaders.
    # Data Loader wraps an iterable over dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Как разделить на Train, Val, Test ???
    # training_data =
    # validation_data =
    # test_data =

    for X, y in dataloader:
    # X, y = next(iter(dataloader))
        print(f"Shape of X [N, C, H, W]: {X.shape}") # torch.Size([64, 1, 28, 28]) torch.float32 [0; 1]
        print(f"Shape of y: {y.shape} {y.dtype}") # torch.Size([64]) torch.int64
        break

    # ---




