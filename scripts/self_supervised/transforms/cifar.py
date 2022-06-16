from kornia import augmentation as augs
from kornia.utils import image_to_tensor
import torch
from torchvision.transforms import RandomApply, ToTensor


IMAGE_SIZE = 32
# todo replace interpolation with bicubic
# from kornia.constants import Resample
# interpolation=Resample.BICUBIC.name


mean_std = {'cifar10': ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            'cifar100': ([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
            'stl10': ([0.43, 0.42, 0.39], [0.27, 0.26, 0.27]),
            'tiny-imagenet': ([0.480, 0.448, 0.398], [0.277, 0.269, 0.282]),
            None: ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])}

def get_cifar_transform(transform_name_list, name=None, image_size=IMAGE_SIZE):
    r"""Transformations used for augmentation. Standard parameters used in SimCLR and BYOL.

    ..note ::
        This function is intended to select a subset of transformations for ablation purposes.
    """
    transforms = []
    if 'crop' in transform_name_list:
        transforms.append(augs.RandomResizedCrop((image_size, image_size), scale=(0.2, 1.0), ratio=(3./4.,4./3.),
                                                 resample=2))  #BICUBIC interpolation
    if 'flip' in transform_name_list:
        transforms.append(augs.RandomHorizontalFlip(p=0.5))
    if 'color_jitter' in transform_name_list:
        transforms.append(RandomApply([augs.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8))
    if 'random_gray' in transform_name_list:
        transforms.append(augs.RandomGrayscale(p=0.1))

    mean, std = mean_std[name]
    transforms.append(augs.Normalize(mean=torch.tensor(mean),
                                     std=torch.tensor(std)))

    return torch.nn.Sequential(*transforms)


def get_cifar_unnormalize(name=None):
    mean, std = mean_std[name]
    return torch.nn.Sequential(
        augs.Normalize(mean=torch.tensor([0., 0., 0.]), std=torch.tensor([1/x for x in std])),
        augs.Normalize(mean=torch.tensor([-x for x in mean]), std=torch.tensor([1., 1., 1.]))
    )


def get_cifar_transform_m(min_scale_factor, horizontal_flip_p, name=None, same_on_batch=False, image_size=IMAGE_SIZE):
    r"""Transformations used for mined views.

    ..note ::
        Currently includes simple spatial transformations.
    """
    transforms = []
    if not (min_scale_factor == 1.0):
        transforms.append(augs.RandomResizedCrop((image_size, image_size), scale=(min_scale_factor, 1.0),
                                                 same_on_batch=same_on_batch))
    if horizontal_flip_p > 0.:
        transforms.append(augs.RandomHorizontalFlip(p=horizontal_flip_p, same_on_batch=same_on_batch))

    mean, std = mean_std[name]
    transforms.append(augs.Normalize(mean=torch.tensor(mean),
                                     std=torch.tensor(std)))
    return torch.nn.Sequential(*transforms)

# Gaussian blur
# from kornia import filters
# filters.GaussianBlur2d((3, 3), (1.5, 1.5))

# def get_cifar_transform(transform_name_list):
#     r"""Transformations used for augmentation. Standard parameters used in SimCLR and BYOL.
#
#     ..note ::
#         This function is intended to select a subset of transformations for ablation purposes.
#     """
#     transforms = []
#     if 'crop' in transform_name_list:
#         transforms.append(torch_transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.2, 1.0)))
#     if 'flip' in transform_name_list:
#         transforms.append(torch_transforms.RandomHorizontalFlip(p=0.5))
#     if 'color_jitter' in transform_name_list:
#         transforms.append(torch_transforms.RandomApply([torch_transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8))
#     if 'random_gray' in transform_name_list:
#         transforms.append(torch_transforms.RandomGrayscale(p=0.2))
#     # todo use imagenet mean and std?
#     transforms.append(torch_transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]))
#
#     return torch_transforms.Compose(transforms)
#
#
# def get_cifar_transform_m(min_scale_factor, horizontal_flip_p):
#     r"""Transformations used for mined views.
#
#     ..note ::
#         Currently includes simple spatial transformations.
#     """
#
#     transforms = []
#     if not (min_scale_factor == 1.0):
#         transforms.append(torch_transforms.RandomResizedCrop(IMAGE_SIZE, scale=(min_scale_factor, 1.0)))
#     if horizontal_flip_p > 0.:
#         transforms.append(torch_transforms.RandomHorizontalFlip(p=0.5))
#     transforms.append(torch_transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]))
#
#     return torch_transforms.Compose(transforms)
