import click
import torch
from torch import nn, optim
from models.model import MyAwesomeModel

import matplotlib.pyplot as plt
import numpy as np
import datetime
import os
from torch.utils.data import DataLoader, TensorDataset


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--epochs", default=5, help="Training epochs")
def train(lr, epochs):
    """Train a model on MNIST."""
    print("Training day and night")
    print("lr: %s\nEpochs: %i" % (str(lr), epochs))

    data_dir = "data/processed/"
    train_images = torch.load(os.path.join(data_dir, "processed_images_train.pt"))
    train_target = torch.load(os.path.join(data_dir, "processed_target_train.pt"))
    # Create datasets
    if len(train_images.shape) == 3:
        train_dataset_tensor = TensorDataset(train_images.unsqueeze(1), train_target)
    else:
        train_dataset_tensor = TensorDataset(train_images, train_target)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset_tensor, batch_size=64, shuffle=True, drop_last=True
    )

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
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        running_loss /= len(train_loader)
        print("Epoch: %i\tLoss: %.3f" % (e + 1, running_loss))

        train_losses.append(running_loss)

    checkpoint_d = {
        "input_channels": input_channels,
        "number_of_kernels": n_kernels,
        "kernel_sizes": kernel_sizes,
        "xsize_input": xsize_input,
        "state_dict": model.state_dict(),
    }
    model_id = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    models_dir = "models/%s/" % model_id

    if not os.path.exists(models_dir):
        os.mkdirs(models_dir)

    torch.save(checkpoint_d, os.path.join(models_dir, "model.pt"))

    plot_dir = "reports/figures/%s/" % model_id
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    plt.plot(np.arange(epochs) + 1, train_losses, "o-")
    plt.xlabel("Epoch")
    plt.ylabel("NLL")
    plt.title("Training loss")
    plt.savefig(os.path.join(plot_dir, "training_loss.png"))
    plt.close()


if __name__ == "__main__":
    train()
