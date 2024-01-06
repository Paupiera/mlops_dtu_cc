import torch.nn as nn
import torch.nn.functional as F


class MyAwesomeModel(nn.Module):

    """My awesome model."""

    def __init__(self, number_of_kernels, kernel_sizes, input_channels, xsize_input):
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

        # self.model = nn.Sequential(
        #     nn.Conv2d(1, 32, 3),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, 3),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(64 * 24 * 24, 10),
        #     nn.LogSoftmax(dim=1),
        # )

    def forward(self, tensor):
        for convlayer in self.convlayers:
            tensor = self.relu(convlayer(tensor))

        output = self.logsoftmax(self.outlinear(self.flattent(tensor)))

        return output
