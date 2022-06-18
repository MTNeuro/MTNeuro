import torch
from torch import nn


from MTNeuro.self_supervised.models.mlp import MLP


class iVAE(nn.Module):
    def __init__(self, latent_dim, data_dim, aux_dim, hidden_dims, aux_disc=True, activation=nn.ReLU(True),
                 decoder_activation=None):
        super().__init__()

        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.aux_dim = aux_dim

        # prior_params
        if not aux_disc:
            self.p = MLP([aux_dim, *hidden_dims, latent_dim], activation=activation)
            self.logl = MLP([aux_dim, *hidden_dims, latent_dim], activation=activation)
        else:
            self.p = nn.Embedding(aux_dim, latent_dim)
            self.logl = nn.Embedding(aux_dim, latent_dim)
        # decoder params
        self.f = MLP([latent_dim, *hidden_dims[::-1], data_dim], activation=activation)
        if decoder_activation is not None:
            self.f = nn.Sequential(self.f, decoder_activation)
        # encoder params
        self.g = MLP([data_dim, *hidden_dims, latent_dim], activation=activation)
        self.logv = MLP([data_dim, *hidden_dims, latent_dim], activation=activation)
        # observation params
        self.obs_log = nn.Linear(1, data_dim, bias=False)

    def encode(self, x):
        return self.g(x), self.logv(x)

    def prior(self, y):
        return self.p(y), self.logl(y)

    def decode(self, z):
        return self.f(z)

    def generate_obs_log_var(self):
        return self.obs_log(torch.FloatTensor([[1]]))

    @staticmethod
    def reparameterize(mu, logvar, mup, logl):
        # q(z) = q(z|x)p(z|u) = N((mu1*var2+mu2*var1)/(var1+var2), var1*var2/(var1+var2));
        mu = mu / (1 + torch.exp(logvar - logl)) + mup / (1 + torch.exp(logl - logvar))
        logvar = logvar + logl - torch.log(torch.exp(logvar) + torch.exp(logl))

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        mu, logvar = self.encode(x)
        mup, logl = self.prior(y)
        z = self.reparameterize(mu, logvar, mup, logl)
        return self.decode(z), mu, logvar, mup, logl
