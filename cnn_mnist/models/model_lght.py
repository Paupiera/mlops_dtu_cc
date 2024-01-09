import os

from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class MyAwesomeModel(LightningModule):

    """My awesome model."""

    def __init__(
        self,
        number_of_kernels=[32, 64],
        kernel_sizes=[3, 3],
        input_channels=1,
        xsize_input=28,
    ):
        super().__init__()

        self.convlayers = nn.ModuleList()
        self.convlayers = nn.ModuleList()

        for n_kernels, kernel_size in zip(number_of_kernels, kernel_sizes):
            self.convlayers.append(nn.Conv2d(input_channels, n_kernels, kernel_size))

            input_channels = n_kernels
            xsize_input -= 2

        self.last_feature_xdim = xsize_input
        self.relu = nn.ReLU()
        self.flattent = nn.Flatten()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.outlinear = nn.Linear(
            number_of_kernels[-1] * self.last_feature_xdim * self.last_feature_xdim, 10
        )

    def forward(self, tensor):
        for convlayer in self.convlayers:
            tensor = self.relu(convlayer(tensor))

        output = self.logsoftmax(self.outlinear(self.flattent(tensor)))

        return output

    def training_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss = nn.functional.nll_loss(output, target.view(-1))

        return loss

    def validation_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        val_loss = nn.functional.nll_loss(output, target.view(-1))
        self.log("val_loss", val_loss)
        return val_loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)


model = MyAwesomeModel()
data_dir = "../data/processed/"
train_images = torch.load(os.path.join(data_dir, "processed_images_train.pt"))
train_target = torch.load(os.path.join(data_dir, "processed_target_train.pt"))
# Create datasets
if len(train_images.shape) == 3:
    train_dataset_tensor = TensorDataset(train_images.unsqueeze(1), train_target)
else:
    train_dataset_tensor = TensorDataset(train_images, train_target)


# Load images
test_images = torch.load(os.path.join(data_dir, "processed_images_test.pt"))
test_target = torch.load(os.path.join(data_dir, "processed_target_test.pt"))

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

# Add callbacks
early_stopping_callback = EarlyStopping(
    monitor="val_loss", patience=1, verbose=True, mode="min", min_delta=0.1
)

trainer = Trainer(
    default_root_dir=os.getcwd(),
    max_epochs=5,
    overfit_batches=0.2,
    callbacks=[early_stopping_callback],
)
trainer.fit(model, train_loader, test_loader)
