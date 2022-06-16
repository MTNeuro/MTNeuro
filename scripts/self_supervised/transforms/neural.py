import torch
import inspect
from collections import namedtuple


class Normalize:
    r"""Normalization transform. Also removes dead neurons

    Args:
        mean (torch.Tensor): Mean.
        std (torch.Tensor): Standard deviation.
    """
    def __init__(self, mean, std):
        if isinstance(mean, float):
            mean = torch.tensor([mean])  # prevent 0 sized tensors

        if isinstance(std, float):
            std = torch.tensor([std])  # prevent 0 sized tensors

        self.not_dead_mask = std != 0
        self.mean = mean[self.not_dead_mask]
        self.std = std[self.not_dead_mask]

    def __call__(self, x, trial=None):
        return (x[:, self.not_dead_mask] - self.mean.to(x)) / self.std.to(x)

    def __repr__(self):
        return '{}(dim={})'.format(self.__class__.__name__, self.mean.size())


class Dropout:
    r"""Drops a neuron with a probability of :obj:`p`. Inplace!

    Args:
        p (float, Optional): Probability of dropout. (default: :obj:`0.5`)
        apply_p (float, Optional): Probability of applying the transformation. (default: :obj:`1.0`)
    """
    def __init__(self, p: float = 0.5, apply_p=1., same_on_trial=True, same_on_batch=False):
        self.p = p
        self.apply_p = apply_p

        assert (not same_on_batch) or (same_on_batch and same_on_trial)
        self.same_on_trial = same_on_trial
        self.same_on_batch = same_on_batch

    def __call__(self, x, trial=None):
        if self.same_on_batch or (trial is None and self.same_on_trial):
            if torch.rand(1) < self.apply_p:
                dropout_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < self.p
                x[:, dropout_mask] = 0

        elif self.same_on_trial:
            dropout_mask = torch.zeros(x.size(), dtype=torch.bool, device=x.device)
            for trial_id in torch.unique(trial):
                if torch.rand(1) < self.apply_p:
                    trial_mask = trial == trial_id
                    dropout_mask[trial_mask] = \
                        torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < self.p
            x[dropout_mask] = 0
        else:
            dropout_mask = torch.empty(x.size(), dtype=torch.float32, device=x.device).uniform_(0, 1) < self.p
            apply_mask = torch.rand(x.size(0)) < 1 - self.apply_p
            dropout_mask[apply_mask] = False
            x[dropout_mask] = 0
        return x

    def __repr__(self):
        return '{}(p={}, apply_p={}, same_on_trial={}, same_on_batch={})'.format(self.__class__.__name__, self.p,
                                                                                 self.apply_p, self.same_on_trial,
                                                                                 self.same_on_batch)


class RandomizedDropout:
    def __init__(self, p: float = 0.5, apply_p=1., same_on_trial=True, same_on_batch=False):
        self.p = p
        self.apply_p = apply_p

        assert (not same_on_batch) or (same_on_batch and same_on_trial)
        self.same_on_trial = same_on_trial
        self.same_on_batch = same_on_batch

    def __call__(self, x, trial=None):
        if self.same_on_batch or (trial is None and self.same_on_trial):
            if torch.rand(1) < self.apply_p:
                p = torch.rand(1) * self.p
                dropout_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < p
                x[:, dropout_mask] = 0

        elif self.same_on_trial:
            dropout_mask = torch.zeros(x.size(), dtype=torch.bool, device=x.device)
            for trial_id in torch.unique(trial):
                if torch.rand(1) < self.apply_p:
                    trial_mask = trial == trial_id
                    p = torch.rand(1) * self.p
                    dropout_mask[trial_mask] = \
                        torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < p
            x[dropout_mask] = 0
        else:
            # generate a random dropout probability for each sample
            p = torch.rand(x.size(0)) * self.p
            # generate dropout mask
            dropout_mask = torch.empty(x.size(), dtype=torch.float32, device=x.device).uniform_(0, 1) < p.view((-1, 1))
            # cancel dropout based on apply probability
            apply_mask = torch.rand(x.size(0)) < 1 - self.apply_p
            dropout_mask[apply_mask] = False
            x[dropout_mask] = 0
        return x

    def __repr__(self):
        return '{}(p={}, apply_p={}, same_on_trial={}, same_on_batch={})'.format(self.__class__.__name__, self.p,
                                                                                 self.apply_p, self.same_on_trial,
                                                                                 self.same_on_batch)


class Noise:
    r"""Adds Gaussian noise to neural activity. The firing rate vector needs to have already been normalized, and
        the Gaussian noise is center and has standard deviation of :obj:`std`.

    Args:
        std (float): Standard deviation of Gaussian noise.
    """
    def __init__(self, std, apply_p=1., same_on_trial=True, same_on_batch=False):
        self.std = std
        self.apply_p = apply_p

        self.same_on_trial = same_on_trial
        self.same_on_batch = same_on_batch

    def __call__(self, x, trial=None):
        if self.same_on_batch or (trial is None and self.same_on_trial):
            if torch.rand(1) < self.apply_p:
                noise = torch.normal(0.0, self.std, size=(x.size(1),), device=x.device)
                x = x + noise
        elif self.same_on_trial:
            for trial_id in torch.unique(trial):
                if torch.rand(1) < self.apply_p:
                    trial_mask = trial == trial_id
                    noise = torch.normal(0.0, self.std, size=(x.size(1),), device=x.device)
                    x[trial_mask] += noise
        else:
            noise = torch.normal(0.0, self.std, size=x.size(), device=x.device)
            # cancel noise based on apply probability
            apply_mask = torch.rand(x.size(0)) < 1 - self.apply_p
            noise[apply_mask] = 0.
            x = x + noise
        return x

    def __repr__(self):
        return '{}(std={}, apply_p={}, same_on_trial={}, same_on_batch={})'.format(self.__class__.__name__, self.std,
                                                                                   self.apply_p, self.same_on_trial,
                                                                                   self.same_on_batch)


class Pepper:
    r"""Adds a constant to the neuron firing rate with a probability of :obj:`p`.

    Args:
        p (float, Optional): Probability of adding pepper. (default: :obj:`0.5`)
        apply_p (float, Optional): Probability of applying the transformation. (default: :obj:`1.0`)
        std (float, Optional): Constant to be added to neural activity. (default: :obj:`1.0`)
    """
    def __init__(self, p=0.5, c=1.0, apply_p=1., same_on_trial=True, same_on_batch=False):
        self.p = p
        self.c = c
        self.apply_p = apply_p

        self.same_on_trial = same_on_trial
        self.same_on_batch = same_on_batch

    def __call__(self, x, trial=None):
        if self.same_on_batch or (trial is None and self.same_on_trial):
            if torch.rand(1) < self.apply_p:
                pepper_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < self.p
                x = x + self.c * pepper_mask
        elif self.same_on_trial:
            for trial_id in torch.unique(trial):
                if torch.rand(1) < self.apply_p:
                    trial_mask = trial == trial_id
                    pepper_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < self.p
                    x[trial_mask] += self.c * pepper_mask
        else:
            pepper_mask = torch.empty(x.size(), dtype=torch.float32, device=x.device).uniform_(0, 1) < self.p
            # cancel pepper based on apply probability
            apply_mask = torch.rand(x.size(0)) < 1 - self.apply_p
            pepper_mask[apply_mask] = False
            x = x + self.c * pepper_mask
        return x

    def __repr__(self):
        return '{}(p={}, c={}, apply_p={}, same_on_trial={}, same_on_batch={})'.format(self.__class__.__name__, self.p,
                                                                                       self.c, self.apply_p,
                                                                                       self.same_on_trial,
                                                                                       self.same_on_batch)

class Compose:
    """Composes several transforms together.

    Args:
        transforms (list of :obj:`transform` objects): List of transforms to
            compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, trial=None):
        for t in self.transforms:
            x = t(x, trial)
        return x

    def __repr__(self):
        args = ['    {},'.format(t) for t in self.transforms]
        return '{}([\n{}\n])'.format(self.__class__.__name__, '\n'.join(args))


def cfg(cls):
    args = inspect.signature(cls).parameters
    fields, defaults = [], []
    for name, param in args.items():
        fields.append(name)
        defaults.append(param.default)
    return namedtuple(cls.__name__+'Cfg', fields, defaults=defaults)


Normalize.cfg = cfg(Normalize)
Dropout.cfg = cfg(Dropout)
RandomizedDropout.cfg = cfg(RandomizedDropout)
Noise.cfg = cfg(Noise)
Pepper.cfg = cfg(Pepper)


def get_neural_transform(*, normalize=None, dropout=None, randomized_dropout=None, noise=None, pepper=None, noise_after_norm=True):
    assert not(dropout is not None and randomized_dropout is not None)

    transforms = []
    if dropout is not None and dropout.p != 0. and dropout.apply_p != 0.:
        transforms.append(Dropout(**dropout._asdict()))
    if randomized_dropout is not None and randomized_dropout.p != 0. and randomized_dropout.apply_p != 0:
        transforms.append(RandomizedDropout(**randomized_dropout._asdict()))
    if pepper is not None and pepper.p != 0 and pepper.c != 0 and pepper.apply_p != 0:
        transforms.append(Pepper(**pepper._asdict()))
    if noise is not None and noise.std !=0 and noise.apply_p !=0:
        transforms.append(Noise(**noise._asdict()))
    if not noise_after_norm:
        transforms.append(Normalize(**normalize._asdict()))
    else:
        transforms.insert(-1, Normalize(**normalize._asdict()))

    transform = Compose(transforms)
    return transform
