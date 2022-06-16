import os

from torchvision.datasets import ImageFolder


def TinyImagenet(root, train=True, **kwargs):
    dataset_path = os.path.join(root, 'train' if train else 'test')
    return ImageFolder(dataset_path, **kwargs)
