import torch
from torch import Tensor
import numpy as np
import torchvision.transforms as transforms
import cv2


class EnhanceEdge(torch.nn.Module):



    def __init__(self, p=0.5, intensity = 1):
        super().__init__()
        
        self.p = p
        self.kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])/8
        self.intensity = intensity
        self.totensor = transforms.ToTensor()

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be enhanced.

        Returns:
            PIL Image or Tensor: Enhanced image.
        """
        if torch.rand(1) < self.p:
            img_e = self.totensor(cv2.filter2D(img.numpy(), -1, self.kernel))*self.intensity + img
            return img_e
        return img


    def __repr__(self):
        return self.__class__.__name__ + "(probability={0}, intensity={1})".format(self.p, self.intensity)
