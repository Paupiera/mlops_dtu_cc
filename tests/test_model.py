import pytest
import torch

from cnn_mnist.models.model import MyAwesomeModel


def test_data():
    n_kernels = [32, 64]
    kernel_sizes = [3, 3]
    input_channels = 1
    xsize_input = 28

    model = MyAwesomeModel(
        number_of_kernels=n_kernels,
        kernel_sizes=kernel_sizes,
        input_channels=input_channels,
        xsize_input=xsize_input,
    )

    image = torch.randn((1, 1, 28, 28))
    log_ps = model(image)

    assert log_ps.shape == (1, 10)


# tests/test_model.py
def test_error_on_wrong_shape():
    n_kernels = [32, 64]
    kernel_sizes = [3, 3]
    input_channels = 1
    xsize_input = 28

    model = MyAwesomeModel(
        number_of_kernels=n_kernels,
        kernel_sizes=kernel_sizes,
        input_channels=input_channels,
        xsize_input=xsize_input,
    )

    with pytest.raises(ValueError, match="Expected input to be a 4D tensor"):
        model(torch.randn(1, 1, 3))


def test_error_on_wrong_n_channels():
    n_kernels = [32, 64]
    kernel_sizes = [3, 3]
    input_channels = 1
    xsize_input = 28

    model = MyAwesomeModel(
        number_of_kernels=n_kernels,
        kernel_sizes=kernel_sizes,
        input_channels=input_channels,
        xsize_input=xsize_input,
    )

    with pytest.raises(ValueError, match="N_channels expected: %i" % input_channels):
        model(torch.randn(5, 2, 28, 28))


def test_error_on_wrong_image_dims():
    n_kernels = [32, 64]
    kernel_sizes = [3, 3]
    input_channels = 1
    xsize_input = 28

    model = MyAwesomeModel(
        number_of_kernels=n_kernels,
        kernel_sizes=kernel_sizes,
        input_channels=input_channels,
        xsize_input=xsize_input,
    )

    with pytest.raises(
        ValueError,
        match="Wrong Input expected dimensions %i %i" % (xsize_input, xsize_input),
    ):
        model(torch.randn(5, 1, 26, 26))


n_kernels = [32, 64]
kernel_sizes = [3, 3]
input_channels = 1
xsize_input = 28


model = MyAwesomeModel(
    number_of_kernels=n_kernels,
    kernel_sizes=kernel_sizes,
    input_channels=input_channels,
    xsize_input=xsize_input,
)


@pytest.mark.parametrize(
    "test_input,error_message",
    [
        ("model(torch.randn(1, 1, 3))", "Expected input to be a 4D tensor"),
        (
            "model(torch.randn(5, 1, 26, 26))",
            "Wrong Input expected dimensions %i %i" % (xsize_input, xsize_input),
        ),
        (
            "model(torch.randn(5, 2, 28, 28))",
            "N_channels expected: %i" % input_channels,
        ),
    ],
)
def test_inputs(test_input, error_message):
    with pytest.raises(
        ValueError,
        match=error_message,
    ):
        eval(test_input)
