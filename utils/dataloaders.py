import torch
from torchvision import datasets, transforms
import os


def get_dataloaders(dataset_name, dataset_root=None, batch_size=8, train=True, transform=None):
    if transform is None:
        transform = transforms.Compose([transforms.Resize(28),  # different img size settings for mnist(28) and svhn(32).
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5,), std=(0.5,))])
    if dataset_root is None:
        dataset_root = r'./datasets/'
        os.makedirs(dataset_root, exist_ok=True)

    dataloaders = None

    if dataset_name == 'mnist':
        dataloaders = get_minist_dataloaders(dataset_root, batch_size, train, transform)
    elif dataset_name == 'svhn':
        dataloaders = get_svhn_dataloaders(dataset_root, batch_size, train, transform)
    else:
        assert 1 == 1, "Given dataset name is not known: {}".format(dataset_name)

    return dataloaders


def get_minist_dataloaders(dataset_root, batch_size, train, transform):
    """Get MNIST datasets loader."""
    # datasets and data loader
    mnist_dataset = datasets.MNIST(root=os.path.join(dataset_root),
                                   train=train,
                                   transform=transform,
                                   download=True)

    if train:
        validation_percentage = 0.1
        valid_image_num = int(len(mnist_dataset) * validation_percentage)
        train_image_num = len(mnist_dataset) - valid_image_num
        train_dataset, valid_dataset = torch.utils.data.random_split(mnist_dataset, [train_image_num, valid_image_num])

        mnist_train_data_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0)

        mnist_validation_data_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0)
        dataloaders = {'train': mnist_train_data_loader, 'validation': mnist_validation_data_loader}

    else:
        mnist_test_data_loader = torch.utils.data.DataLoader(
            dataset=mnist_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0)
        dataloaders = {'test': mnist_test_data_loader}

    return dataloaders


def get_svhn_dataloaders(dataset_root, batch_size, train, transform):
    """Get SVHN datasets loader."""
    # datasets and data loader
    if train:
        svhn_dataset = datasets.SVHN(root=os.path.join(dataset_root), split='train', transform=transform, download=True)
        svhn_dataset_extra = datasets.SVHN(root=os.path.join(dataset_root), split='extra', transform=transform, download=True)
        validation_percentage = 0.1
        valid_image_num = int(len(svhn_dataset) * validation_percentage)
        train_image_num = len(svhn_dataset) - valid_image_num
        train_dataset, valid_dataset = torch.utils.data.random_split(svhn_dataset, [train_image_num, valid_image_num])
        svhn_train_data_loader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.ConcatDataset((train_dataset, svhn_dataset_extra)),
            batch_size=batch_size,
            shuffle=True, drop_last=True)

        svhn_validation_data_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size,
                                                                  shuffle=True, drop_last=True)

        dataloaders = {'train': svhn_train_data_loader, 'validation': svhn_validation_data_loader}
    else:
        svhn_test_dataset = datasets.SVHN(root=os.path.join(dataset_root), split='test', transform=transform, download=True)
        svhn_test_data_loader = torch.utils.data.DataLoader(dataset=svhn_test_dataset, batch_size=batch_size,
                                                            shuffle=True, drop_last=True)
        dataloaders = {'test': svhn_test_data_loader}

    return dataloaders


# # Test local functions
# if __name__ == '__main__':
#
#     mnist_dataloaders = get_dataloaders('mnist', batch_size=8, train=True)
#     svhn_dataloaders = get_dataloaders('svhn', batch_size=8, train=True)
#
#     mnist_test_dataloaders = get_dataloaders('mnist', batch_size=8, train=False)
#     svhn_test_dataloaders = get_dataloaders('svhn', batch_size=8, train=False)
#
#     tmp = 0