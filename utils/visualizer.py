import numpy as np

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import tkinter
# tkinter._test()
# print(matplotlib.get_backend())

import torch

def plot(entries, xlabel='epoch', ylabel='', title=''):
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(entries)), entries)
    ax.set(xlabel='epoch', ylabel=ylabel, title=title)
    ax.grid()
    # fig.savefig("test.png")
    plt.show()

# plot(train_losses, ylabel='loss')
# plot(train_accuracies, ylabel='accuracy')
# plot(test_losses, ylabel='loss')
# plot(test_accuracies, ylabel='accuracy')

def plot4(trloss, tracc, tloss, tacc):
    fig, axs = plt.subplots(2, 2)
    axs[0][0].plot(np.arange(len(trloss)), trloss)
    axs[0][1].plot(np.arange(len(tracc)), tracc)
    axs[1][0].plot(np.arange(len(tloss)), tloss)
    axs[1][1].plot(np.arange(len(tacc)), tacc)
    #
    axs[0][0].grid()
    axs[0][1].grid()
    axs[1][0].grid()
    axs[1][1].grid()
    #
    axs[0][0].set(xlabel='epoch', ylabel="loss", title="Train loss")
    axs[0][1].set(xlabel='epoch', ylabel="accuracy", title="Train accuracy")
    axs[1][0].set(xlabel='epoch', ylabel="loss", title="Test loss")
    axs[1][1].set(xlabel='epoch', ylabel="accuracy", title="Test accuracy")
    # fig.savefig("test.png")
    plt.show()


def visualize_image_dataset(dataset, classes):
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