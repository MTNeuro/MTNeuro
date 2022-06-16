from kornia import augmentation as augs
from kornia.geometry.transform import affine
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, RandomCrop, RandomErasing, RandomApply, ColorJitter, GaussianBlur, RandomAffine
from .RandErasingMean import RandErasingMean
from .CustomAugs import RandomLocalCutMix, RandomCropS, RandomCropSlice
import numpy as np

IMAGE_SIZE = 28
# todo replace interpolation with bicubic
# from kornia.constants import Resample
# interpolation=Resample.BICUBIC.name


mean_std = {'synapse': ([0.5098], [0.2376]),
            'path': ([0.5], [0.5]),
            None: ([0.5], [0.2])}
            
class RandSlice(torch.nn.Module):

    def __init__(self, p=1.):
        super().__init__()

        self.p = p

    def forward(self, img):

        if torch.rand(1) <= self.p:
        
            s_img = torch.zeros_like(img[..., 0, :,:])
            for i in range(img.shape[0]):
                s = np.random.randint(0,28)

                s_img[i,:,:,:] = img[i, :, s, :,:]
        return s_img.float()
                       
            
class SynapseAugmentationPipeline(nn.Module):
    #todo: improve efficiency by removing conditionals from forward pass
    def __init__(self, transform_name_list, name=None, crop_size=18, image_size=IMAGE_SIZE, p = 0.5, same_on_batch=True):
        super(SynapseAugmentationPipeline, self).__init__()
        self.slice = RandSlice(p=1)
        self.cropslice = RandomCropSlice(height=crop_size, width=crop_size, depth=crop_size, full_h = image_size, full_w = image_size, full_d = image_size, same_on_batch = same_on_batch)
        self.crp = augs.RandomCrop((crop_size, crop_size), same_on_batch=same_on_batch)
        mean, std = mean_std[name]
        self.normalize = augs.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        self.transform_list = transform_name_list
        
    def forward(self, input, label):
    
        
        if 'random_crop_slice' in self.transform_list:
            input = input[:,0,...]
            input, _ = self.cropslice(input, label)
        if 'random_slice' in self.transform_list:
            input = self.slice(input)
        if 'random_crop' in self.transform_list:
            input = self.crp(input) 
        input = self.normalize(input)
        return input, label
        

def get_synapse_transform(transform_name_list, name=None, crop_size=18, image_size=IMAGE_SIZE, p = 0.5, same_on_batch=True):
    r"""Transformations used for augmentation. Standard parameters used in SimCLR and BYOL.

    ..note ::
        This function is intended to select a subset of transformations for ablation purposes.
    """
    
    transforms= SynapseAugmentationPipeline(transform_name_list, name, crop_size, image_size, p, same_on_batch)
    return transforms
    

def get_synapse_unnormalize(name=None):
    mean, std = mean_std[name]
    return torch.nn.Sequential(
        augs.Normalize(mean=torch.tensor([0.]), std=torch.tensor([1/x for x in std])),
        augs.Normalize(mean=torch.tensor([-x for x in mean]), std=torch.tensor([1.]))
    )


def get_synapse_transform_m(name=None, image_size=IMAGE_SIZE):
    r"""Transformations used for mined views.
    ..note ::
        Currently includes simple spatial transformations.
    """
    transforms = []
    transforms.append(RandSlice(p=1))

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

