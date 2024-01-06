import click
import torch
from models.model import MyAwesomeModel
from torch.utils.data import DataLoader

import os


@click.command()
@click.argument("model_checkpoint")
@click.argument("data_file")
def predict(model_checkpoint, data_file):
    """Evaluate a trained model."""

    print("Evaluating like my life dependends on it")

    checkpoint = torch.load(model_checkpoint)
    model = MyAwesomeModel(
        checkpoint["number_of_kernels"],
        checkpoint["kernel_sizes"],
        checkpoint["input_channels"],
        checkpoint["xsize_input"],
    )
    model.load_state_dict(checkpoint["state_dict"])

    # Load images
    test_images = torch.load(data_file)

    # Create datasets
    if len(test_images.shape) == 3:
        test_images = test_images.unsqueeze(1)

    # Create dataloaders
    test_loader = DataLoader(test_images, batch_size=64, shuffle=True, drop_last=True)

    ## TODO: Implement the validation pass and print out the validation accuracy
    with torch.no_grad():
        model.eval()
        return [model(images).exp().topk(k=1)[1] for images in test_loader]


if __name__ == "__main__":
    predict()
