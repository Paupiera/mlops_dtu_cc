import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim

from models.model import MyAwesomeModel


def train_cml(train_loader, test=False):
    """Train a model on MNIST."""
    print("Training day and night CML")

    input_channels, xsize_input = next(iter(train_loader))[0].shape[1:3]

    n_kernels = [32, 64]
    kernel_sizes = [3, 3]

    model = MyAwesomeModel(
        number_of_kernels=n_kernels,
        kernel_sizes=kernel_sizes,
        input_channels=input_channels,
        xsize_input=xsize_input,
    )
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 5
    model.train()
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        model.eval()
        preds, target = [], []
        for images, labels in test_loader:
            log_ps = model(images)

            probs = torch.exp(log_ps)
            preds.append(probs.argmax(dim=-1))

            target.append(labels)

    target = torch.cat(target, dim=0)
    preds = torch.cat(preds, dim=0)

    model_id = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    plot_dir = "reports/figures/%s/" % model_id
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    report = classification_report(target, preds)
    with open(os.path.join(plot_dir, "classification_report.txt"), "w") as outfile:
        outfile.write(report)
    confmat = confusion_matrix(target, preds)
    print(confmat)
    disp = ConfusionMatrixDisplay(confmat)
    disp.plot()
    plt.savefig(os.path.join(plot_dir, "confusion_matrix.png"))


if __name__ == "__main__":
    data_dir = "data/processed/"
    train_images = torch.load(os.path.join(data_dir, "processed_images_train.pt"))
    train_target = torch.load(os.path.join(data_dir, "processed_target_train.pt"))
    print(train_target.shape)
    # Create datasets
    if len(train_images.shape) == 3:
        train_dataset_tensor = TensorDataset(train_images.unsqueeze(1), train_target)
    else:
        train_dataset_tensor = TensorDataset(train_images, train_target)

    test_images = torch.load(os.path.join(data_dir, "processed_images_test.pt"))
    test_target = torch.load(os.path.join(data_dir, "processed_target_test.pt"))
    print(test_target.shape)
    # Create datasets
    if len(test_images.shape) == 3:
        test_dataset_tensor = TensorDataset(test_images.unsqueeze(1), test_target)
    else:
        test_dataset_tensor = TensorDataset(test_images, test_target)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset_tensor, batch_size=64, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset_tensor, batch_size=64, shuffle=False, drop_last=False
    )

    train_cml(train_loader, test_loader)
