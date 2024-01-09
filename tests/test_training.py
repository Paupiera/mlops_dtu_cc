import os

from tests import _PROJECT_ROOT
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from cnn_mnist.train_model import train


def test_training():
    X = torch.randn((64, 1, 28, 28))
    Y = torch.randint(high=10, size=(64,))
    train_dataset_tensor = TensorDataset(X, Y)
    train_loader = DataLoader(train_dataset_tensor, batch_size=1, shuffle=True)
    e = train(train_loader, epochs=5, test=True)
    assert e == 4, "Training did not finish"
