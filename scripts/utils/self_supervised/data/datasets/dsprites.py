from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import os
from torchvision import transforms


class DSprites(Dataset):
    latent_factor_map = {'shape': 1, 'scale': 2, 'orientation':3, 'posX':4, 'posY':5}
    def __init__(self, root, label='all', downsample_data_rate=None, downsample_latent_rates=None):
        self.root = root
        self.label = label
        self.downsample_data_rate = downsample_data_rate
        self.downsample_latent_rates = downsample_latent_rates
        self.data, self.latents_values, self.metadata = self._load_data()

        if self.downsample_data_rate is not None:
            self._downsample_data()

        if self.downsample_latent_rates is not None:
            self._downsample_latent()

    def _load_data(self):
        root = os.path.join(self.root, 'dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        data = np.load(root, encoding='bytes', allow_pickle=True)
        imgs = torch.from_numpy(data['imgs']).unsqueeze(1).float()
        latents_values = torch.from_numpy(data['latents_values']).float()
        latents_classes = torch.from_numpy(data['latents_classes'])
        metadata = data['metadata']
        return imgs, latents_values, metadata

    def _downsample_data(self, seed=0):
        keep_size = int(np.ceil(len(self) * self.downsample_data_rate))
        np.random.seed(seed)
        keep_mask = np.random.choice(len(self), keep_size)
        self.data = self.data[keep_mask]
        self.latents_values = self.latents_values[keep_mask]

    def _downsample_latent(self, seed=0):
        np.random.seed(seed)
        for latent_factor, downsample_rate in self.downsample_latent_rates.items():
            latent_id = self.latent_factor_map[latent_factor]
            unique_values = np.unique(self.latents_values[:, latent_id])
            keep_size = int(np.ceil(len(unique_values) * downsample_rate))
            keep_latent_values = np.random.choice(len(unique_values), keep_size)
            print(latent_factor, ':', unique_values[keep_latent_values])
            keep_samples_mask = torch.ones(self.latents_values.size(0), dtype=bool)
            for latent_value in unique_values[keep_latent_values]:
                keep_samples_mask &= self.latents_values[:, latent_id]!=latent_value
            self.data = self.data[keep_samples_mask]
            self.latents_values = self.latents_values[keep_samples_mask]

    def __getitem__(self, index):
        if self.label == 'all':
            return self.data[index], {'latents_values':self.latents_values[index],
                                      'latents_classes': self.latents_values[index]}
        elif self.label == 'shape':
            return self.data[index], self.latents_values[index, 1].long()

    def __len__(self):
        return self.data.size(0)

    @staticmethod
    def prepare_views(inputs):
        x, labels = inputs
        outputs = {'view1': x, 'view2': x}
        return outputs

    def random_crop_transform(self, img):
        pad = transforms.Pad(6, fill=0, padding_mode='constant')
        crop = transforms.RandomResizedCrop(64, scale=(0.6, 0.9),ratio=(1., 1.))
