import torch.nn as nn


class MyAwesomeModel(nn.Module):

    """My awesome model."""

    def __init__(self, number_of_kernels, kernel_sizes, input_channels, xsize_input):
        super().__init__()

        self.convlayers = nn.ModuleList()
        self.convlayers = nn.ModuleList()
        self.input_channels = input_channels
        self.xsize_input = xsize_input

        for n_kernels, kernel_size in zip(number_of_kernels, kernel_sizes):
            self.convlayers.append(nn.Conv2d(input_channels, n_kernels, kernel_size))

            input_channels = n_kernels
            xsize_input -= 2

        self.last_feature_xdim = xsize_input
        self.relu = nn.ReLU()
        self.flattent = nn.Flatten()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.outlinear = nn.Linear(number_of_kernels[-1] * self.last_feature_xdim * self.last_feature_xdim, 10)

    def forward(self, tensor):
        if tensor.ndim != 4:
            raise ValueError("Expected input to be a 4D tensor")

        if tensor[0].shape[0] != self.input_channels:
            raise ValueError("N_channels expected: %i" % self.input_channels)
        if tensor[0].shape[1:] != (self.xsize_input, self.xsize_input):
            raise ValueError("Wrong Input expected dimensions %i %i" % (self.xsize_input, self.xsize_input))

        for convlayer in self.convlayers:
            tensor = self.relu(convlayer(tensor))

        output = self.logsoftmax(self.outlinear(self.flattent(tensor)))

        return output
