from kornia import augmentation as augs
from kornia.geometry.transform import affine
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, RandomCrop, RandomErasing, RandomApply, ColorJitter, GaussianBlur, RandomAffine
from .RandErasingMean import RandErasingMean
from .CustomAugs import RandomLocalCutMix, RandomCropS, RandomCropSlice


IMAGE_SIZE = 32
# todo replace interpolation with bicubic
# from kornia.constants import Resample
# interpolation=Resample.BICUBIC.name


mean_std = {'xray': ([0.4894], [0.1238]),
            None: ([0.5], [0.2])}
            
class XrayAugmentationPipeline(nn.Module):
    #todo: improve efficiency by removing conditionals from forward pass
    def __init__(self, transform_name_list, name=None, image_size=IMAGE_SIZE, p = 0.5, same_on_batch=False, dataset_type="2D"):
        super(XrayAugmentationPipeline, self).__init__()
        self.mixup = augs.RandomMixUp(p=p, lambda_val=(0,0.5))
        self.zoom = RandomApply([RandomAffine(degrees=0, scale=(1,2))], p = p)
        self.sharpen = augs.RandomSharpness(p=p)
        self.crp = RandomCrop(image_size) #augs.RandomCrop((image_size, image_size), same_on_batch=same_on_batch)
        self.cropslice = RandomCropSlice(height=image_size,width=image_size,depth=10, same_on_batch=same_on_batch)
        self.hflip = augs.RandomHorizontalFlip(p=p)
        self.vflip = augs.RandomVerticalFlip(p=p)
        self.rotate = augs.RandomRotation(p=p, degrees=45)
        self.solarize = augs.RandomSolarize(p=p)
        self.cutmix = augs.RandomCutMix(32,32, p=p)
        self.loccutmix = RandomLocalCutMix(64,64, cut_size = torch.tensor([0.15, 0.45]), p=p, dataset_type=dataset_type)
        self.erasemean = RandErasingMean(p=p, scale=(0.1,0.1))
        self.erase = RandomErasing(p=p, scale=(0.1,0.1))
        self.center_crop = augs.CenterCrop(size=image_size)
        mean, std = mean_std[name]
        self.normalize = augs.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        self.transform_list = transform_name_list
        
    def forward(self, input, label):
        
        if 'random_zoom' in self.transform_list:
            input = self.zoom(input)
        if 'random_mixup' in self.transform_list:
            input, label = self.mixup(input, label)
        if 'random_cutmix' in self.transform_list:
            input, label = self.cutmix(input, label) 
        if 'random_localcutmix' in self.transform_list:
            input, _ = self.loccutmix(input, label)
        if 'random_hflip' in self.transform_list:
            input = self.hflip(input)  
        if 'random_vflip' in self.transform_list:
            input = self.vflip(input)
        if 'random_crop_slice' in self.transform_list:
            input, _ = self.cropslice(input, label)
        if 'random_rotate' in self.transform_list:
            input = self.center_crop(self.rotate(input))
        elif 'random_crop' in self.transform_list:
            input = self.crp(input) 
        if 'random_solarize' in self.transform_list:
            input = self.solarize(input)
        if 'random_sharpen' in self.transform_list:
            input = self.sharpen(input)
        if 'random_erase' in self.transform_list:
            input = self.erase(input)
        if 'random_erase_mean' in self.transform_list:
            input = self.erasemean(input)
        input = self.normalize(input)
        return input, label
        
        
class XrayAugmentationPipelineSeg(nn.Module):
    #todo: improve efficiency by removing conditionals from forward pass
    def __init__(self, transform_name_list, name=None, image_size=IMAGE_SIZE, p = 0.5):
        super(XrayAugmentationPipelineSeg, self).__init__()
        self.crp = augs.RandomCrop(size=(image_size,image_size), return_transform=True)
        self.affine = affine
        self.loccutmix = RandomLocalCutMix(32,32, cut_size = torch.tensor([0.15, 0.45]), p=p, transf_labels=True)

        mean, std = mean_std[name]
        self.normalize = augs.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        self.transform_list = transform_name_list
        
    def forward(self, input, label):
        
        if 'random_localcutmix' in self.transform_list:
            input, label = self.loccutmix(input, label)
        if 'random_crop' in self.transform_list:
            input, transform = self.crp(input) 
            label = self.affine(label, transform)
        input = self.normalize(input)
        return input, label

def get_xray_transform(transform_name_list, name=None, image_size=IMAGE_SIZE, p = 0.5, same_on_batch=False, dataset_type="2D"):
    r"""Transformations used for augmentation. Standard parameters used in SimCLR and BYOL.

    ..note ::
        This function is intended to select a subset of transformations for ablation purposes.
    """
    
    transforms= XrayAugmentationPipeline(transform_name_list, name, image_size, p, same_on_batch, dataset_type)
    return transforms
    
def get_xray_transform_seg(transform_name_list, name=None, image_size=IMAGE_SIZE, p = 0.5):
    r"""Transformations used for augmentation. Standard parameters used in SimCLR and BYOL.

    ..note ::
        This function is intended to select a subset of transformations for ablation purposes.
    """
    
    transforms= XrayAugmentationPipelineSeg(transform_name_list, name, image_size, p)
    return transforms


def get_xray_unnormalize(name=None):
    mean, std = mean_std[name]
    return torch.nn.Sequential(
        augs.Normalize(mean=torch.tensor([0.]), std=torch.tensor([1/x for x in std])),
        augs.Normalize(mean=torch.tensor([-x for x in mean]), std=torch.tensor([1.]))
    )


def get_xray_transform_m(m_randcrop, horizontal_flip_p, name=None, same_on_batch=False, image_size=IMAGE_SIZE, cr_type='k'):
    r"""Transformations used for mined views.
    ..note ::
        Currently includes simple spatial transformations.
    """
    transforms = []
    if m_randcrop and cr_type=='k':
        transforms.append(augs.RandomCrop((image_size, image_size), same_on_batch=same_on_batch))
    elif m_randcrop and cr_type=='t':
        transforms.append(RandomCrop(image_size))
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

