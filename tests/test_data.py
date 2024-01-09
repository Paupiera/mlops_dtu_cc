import os
import os.path
import pytest

from tests import _PATH_DATA
import torch


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data():
    data_dir = os.path.join(_PATH_DATA, "processed/")

    train_images = torch.load(os.path.join(data_dir, "processed_images_train.pt"))
    train_target = torch.load(os.path.join(data_dir, "processed_target_train.pt"))

    test_images = torch.load(os.path.join(data_dir, "processed_images_test.pt"))
    test_target = torch.load(os.path.join(data_dir, "processed_target_test.pt"))

    N_train, N_test = 30000, 5000

    assert len(train_target) == N_train and len(test_target) == N_test, "Targets/Images dimensions do not match"

    assert train_images.shape == (len(train_target), 28, 28) and test_images.shape == (
        len(test_target),
        28,
        28,
    )

    assert (
        len(set(train_target.tolist())) == len(set(test_target.tolist())) == 10
    ), "Not all classess are represented on the train test datasets"
