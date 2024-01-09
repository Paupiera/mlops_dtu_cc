# import click
import torch
from torch import nn, optim
from cnn_mnist.models.model import MyAwesomeModel

import matplotlib.pyplot as plt
import numpy as np
import datetime
import os
from torch.utils.data import DataLoader, TensorDataset


def train(train_loader, lr=1e-3, epochs=5, test=False):
    """Train a model on MNIST."""
    print("Training day and night")
    print("lr: %s\nEpochs: %i\nTest: %s" % (str(lr), epochs, test))

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
        os.makedirs(models_dir)
    if not test:
        torch.save(checkpoint_d, os.path.join(models_dir, "model.pt"))

    plot_dir = "reports/figures/%s/" % model_id
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plt.plot(np.arange(epochs) + 1, train_losses, "o-")
    plt.xlabel("Epoch")
    plt.ylabel("NLL")
    plt.title("Training loss")
    if not test:
        plt.savefig(os.path.join(plot_dir, "training_loss.png"))
    plt.close()

    return e


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

    # Create dataloaders
    train_loader = DataLoader(train_dataset_tensor, batch_size=64, shuffle=True, drop_last=True)

    train(train_loader)
