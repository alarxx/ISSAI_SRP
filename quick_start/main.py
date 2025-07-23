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
from torchvision import datasets
from torchvision.transforms import ToTensor

import torch.nn.functional as F
import torch.optim as optim

# ---

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(), # samples
    # target_transform # labels
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(), # samples
    # target_transform # labels
)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

# ---

def visualize(dataset=training_data):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

# visualize(training_data)
# visualize(test_data)

# ---

batch_size = 64

# Create data loaders.
# Data Loader wraps an iterable over dataset
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
# X, y = next(iter(test_dataloader))
    print(f"Shape of X [N, C, H, W]: {X.shape}") # torch.Size([64, 1, 28, 28]) torch.float32 [0; 1]
    print(f"Shape of y: {y.shape} {y.dtype}") # torch.Size([64]) torch.int64
    break

# ---

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() # flattens an image to row-vector [<->], so rows are samples -> shape[n, 784]
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def __setattr__(self, name, value):
        print(f"__setattr__ called: {name} = {value}")
        super().__setattr__(name, value)

model = NeuralNetwork().to(device)
# print(type(nn.Sequential()))
# print(type(nn.Module()))
# print(type(model))
# print(model)
# print(model.parameters())
# for param in model.parameters():
#     print(param) # tensors
print(f"Model structure: {model}\n\n")
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values[:2] : {param[:2]} \n")

X = torch.rand(2, 1, 28, 28, device=device) # (N, C, H, W)
logits = model(X)
print(logits.shape)
pred_probab = nn.Softmax(dim=1)(logits) # along row, columns are entries
print(pred_probab)
y_pred = pred_probab.argmax(dim=1) # along row
# y_pred = [y1, y2]
print(f"Predicted class: {y_pred}")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad(): # temporarily changing global state - torch.is_grad_enabled()
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

# ---

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))

print(f"Test loaded model\n-------------------------------")
test(test_dataloader, model, loss_fn)

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = labels_map[pred[0].argmax(0).item()], labels_map[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
