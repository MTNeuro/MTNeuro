import random

import torch
import numpy as np


def set_random_seeds(random_seed=0):
    r"""Sets the seed for generating random numbers.

    Args:
        random_seed: Desired random seed.
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms(True)
    # todo remove this improves performance
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

"""
From Pytorch lightning's seed_everything
    log.info(f"Global seed set to {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
"""
