import numpy as np

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import tkinter

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
