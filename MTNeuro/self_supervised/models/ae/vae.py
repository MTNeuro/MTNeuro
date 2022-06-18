import torch
from torch import nn

from MTNeuro.self_supervised.models.mlp import MLP

class VAE(nn.Module):
    def __init__(self, latent_dim, data_dim, hidden_dims, activation=nn.ReLU(True)):
        super().__init__()

        self.data_dim = data_dim
        self.latent_dim = latent_dim

        # decoder params
        self.f = MLP([latent_dim, *hidden_dims[::-1], data_dim], activation=activation)
        # encoder params
        self.g = MLP([data_dim, *hidden_dims, latent_dim], activation=activation)
        self.logv = MLP([data_dim, *hidden_dims, latent_dim], activation=activation)

    def encode(self, x):
        return self.g(x), self.logv(x)

    def decode(self, z):
        return self.f(z)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
