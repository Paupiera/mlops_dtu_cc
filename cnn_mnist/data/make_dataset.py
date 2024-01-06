if __name__ == "__main__":
    # Get the data and process it
    import torch
    import os

    def standarize(tensor):
        mean = tensor.mean(dim=(1, 2))
        std = tensor.std(dim=(1, 2))

        return (tensor - mean[:, None, None]) / std[:, None, None]

    # data_dir = "/Users/mbv396/PhD/Courses/dtu_mlops/data/corruptmnist/"
    data_dir = "data/raw/corruptmnist/"

    # load train images and targets and aggreagte into tensor
    for i in range(6):
        train_images_file = "train_images_%i.pt" % i
        train_target_file = "train_target_%i.pt" % i
        images_i = torch.load(os.path.join(data_dir, train_images_file))
        target_i = torch.load(os.path.join(data_dir, train_target_file))

        if i == 0:
            train_images = images_i
            train_target = target_i
        else:
            train_images = torch.cat((train_images, images_i), dim=0)
            train_target = torch.cat((train_target, target_i), dim=0)

    # load test images and targets
    test_images = torch.load(os.path.join(data_dir, "test_images.pt"))
    test_target = torch.load(os.path.join(data_dir, "test_target.pt"))

    # apply transformation to images
    train_images_stz = standarize(train_images)
    test_images_stz = standarize(test_images)

    # Save test images
    data_dir = "data/processed/"
    torch.save(train_images_stz, os.path.join(data_dir, "processed_images_train.pt"))
    torch.save(test_images_stz, os.path.join(data_dir, "processed_images_test.pt"))

    torch.save(train_target, os.path.join(data_dir, "processed_target_train.pt"))
    torch.save(test_target, os.path.join(data_dir, "processed_target_test.pt"))
