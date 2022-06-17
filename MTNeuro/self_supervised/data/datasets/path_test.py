import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from PIL import ImageFilter
import cv2 
import os
from os.path import expanduser
from einops import rearrange


class PathTest(Dataset):
    
    def __init__(self, size=14):
        path  = os.path.join(expanduser("~"), ".medmnist/pathmnist.npz")
        npz_file = np.load(path)
        self.labels = npz_file['test_labels']
        self.imgs = npz_file['test_images']
        self.size = size
        self.num_samples = self.imgs.shape[0]*4

    def __getitem__(self, key):
        slice_id, corner_id = divmod(key, 4)
        img, label = self.imgs[slice_id], self.labels[slice_id].astype(int)
        img = np.stack([img/255.], axis=0)
        size = self.size
       
        if corner_id==0:
            crop = img[0, :size, :size, :]
        elif corner_id==1:
            crop = img[0, :size, -size:, :]
        elif corner_id==2:
            crop = img[0, -size:, :size, :]
        else:
            crop = img[0, -size:, -size:, :]    
        return np.moveaxis(crop, -1, 0), label
            
    def __len__(self):
        return self.num_samples
        
