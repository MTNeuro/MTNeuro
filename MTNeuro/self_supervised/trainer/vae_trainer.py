import logging

import torch

from MTNeuro.self_supervised.trainer import Trainer
from MTNeuro.self_supervised.loss import reconstruction_loss_w_obs_noise, poisson_reconstruction_loss, i_kl_divergence


log = logging.getLogger(__name__)


class iVAETrainer(Trainer):
    def __init__(self,
                 *,
                 model,
                 poisson=False,
                 train_dataloader,
                 lr,
                 no_cuda=False,
                 gpu=None,
                 log_steps,
                 logdir,
                 ):
        super().__init__(distributed=False, no_cuda=no_cuda, gpu=gpu, world_size=1,
                         rank=0, log_steps=log_steps, logdir=logdir)

        # model
        self.model = model.to(self.device)
        self.poisson = poisson

        # data
        self.train_dataloader = train_dataloader

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def train_epoch(self):
        self.model.train()

        for x, y in self.train_dataloader:
            # get inputs
            x, y = x.to(self.device), y.to(self.device)

            # forward
            self.optimizer.zero_grad()
            recon_x, mu, logvar, mup, logl = self.model(x, y)

            # compute kl divergence
            kl_div = i_kl_divergence(mu, logvar, mup, logl)

            # compute reconstruction loss
            if not self.poisson:
                obs_log_var = self.model.generate_obs_log_var()
                recon_loss = reconstruction_loss_w_obs_noise(recon_x, x, obs_log_var)
            else:
                recon_x = torch.clip(recon_x, min=1e-7, max=1e7)
                recon_loss = poisson_reconstruction_loss(recon_x, x)

            loss = kl_div + recon_loss

            loss.backward()

            # update
            self.optimizer.step()

            # log scalars
            if self.rank == 0 and self.step % self.log_steps == 0:
                self.log_scalars(loss=loss.item())

            # update step
            self.step += 1
        self.epoch += 1
