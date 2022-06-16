import torch
from torchvision import transforms
import cv2
from PIL import Image, ImageOps
import numpy as np
from torchvision import datasets


class GaussianBlur():
    def __init__(self, kernel_size, sigma_min=0.1, sigma_max=2.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.kernel_size = kernel_size

    def __call__(self, img):
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        img = cv2.GaussianBlur(np.array(img), (self.kernel_size, self.kernel_size), sigma)
        return Image.fromarray(img.astype(np.uint8))

class Solarize():
    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, sample):
        return ImageOps.solarize(sample, self.threshold)


class ImageNet:
    def __init__(self, root, split, transform_list, label=False):
        self.dataset = datasets.ImageNet(root, split=split)
        self.transform_list = transform_list
        self.label = label

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        views = [transform(img).unsqueeze(0) for transform in self.transform_list]
        if self.label:
            views.append(torch.tensor(label, dtype=torch.long))
        return views

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def prepare_views(inputs):
        view1, view2 = inputs
        view1 = torch.squeeze(view1)
        view2 = torch.squeeze(view2)
        outputs = {'view1': view1, 'view2': view2}
        return outputs



color_jitter = transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform_1 = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([color_jitter], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur(kernel_size=23)], p=1.0),
    transforms.RandomApply([Solarize()], p=0.0),
    transforms.ToTensor(),
    normalize
])

transform_2 = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([color_jitter], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur(kernel_size=23)], p=0.1),
    transforms.RandomApply([Solarize()], p=0.2),
    transforms.ToTensor(),
    normalize
])

transform_myow = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomApply([GaussianBlur(kernel_size=23)], p=0.5),
    transforms.ToTensor(),
    normalize
])

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])
