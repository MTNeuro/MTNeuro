import os
from contextlib import nullcontext
import logging
from functools import partial

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD, Adam, AdamW
from torch.nn.parallel import DistributedDataParallel as DDP

from MTNeuro.self_supervised.optim import LARS
from MTNeuro.self_supervised.utils import you_only_log_once


from abc import ABC

log = logging.getLogger(__name__)

# todo improve logging, limit it to rank 0 gpu when it's possible.


class ScheduleExtension:
    r"""Allows scheduling."""
    def _cste_warmup(self, step, max_val, warmup_steps):
        assert 0 <= step <= warmup_steps
        return max_val

    def _linear_warmup(self, step, max_val, warmup_steps):
        assert 0 <= step <= warmup_steps
        return max_val * step / warmup_steps

    def _cosine_decay(self, step, max_val, total_steps, warmup_steps=0, **decay_params):
        assert warmup_steps <= step <= total_steps
        return max_val * (1 + np.cos((step - warmup_steps) * np.pi / (total_steps - warmup_steps))) / 2

    def _poly_decay(self, step, max_val, total_steps, warmup_steps=0, **decay_params):
        assert 'decay_n' in decay_params
        assert warmup_steps <= step <= total_steps
        return max_val * (1 - ((step - warmup_steps) / (total_steps - warmup_steps)) ** decay_params['decay_n'])

    def _step_decay(self, step, max_val, total_steps, warmup_steps=0, **decay_params):
        assert 'decay_milestones' in decay_params and 'decay_gamma' in decay_params
        assert warmup_steps <= step <= total_steps
        for i, milestone in enumerate(decay_params['decay_milestones']):
            if step < milestone:
                break
        return max_val * (decay_params['decay_gamma'] ** i)

    def _decay(self, step, max_val, total_steps,
               decay_type='cosine', **decay_parmas):
        decay_func = {'cosine': self._cosine_decay, 'poly': self._poly_decay}[decay_type]
        return decay_func(step, max_val, total_steps=total_steps, **decay_parmas)

    def _warmup_then_decay(self, step, max_val, warmup_steps, total_steps, warmup_type='linear',
                           decay_type='cosine', **decay_parmas):
        """learning rate warm up and decay"""
        if step < warmup_steps:
            warmup_func = {'linear': self._linear_warmup, 'cste': self._cste_warmup}[warmup_type]
            return warmup_func(step, max_val, warmup_steps)
        elif warmup_steps <= step <= total_steps:
            decay_func = {'cosine': self._cosine_decay, 'poly': self._poly_decay,
                          'step': self._step_decay}[decay_type]
            return decay_func(step, max_val, warmup_steps=warmup_steps, total_steps=total_steps, **decay_parmas)
        else:
            raise ValueError('{}/{}'.format(step, warmup_steps))


class OptimizerSelectExtension:
    r"""Allows to select optimizer."""
    __optimizer_dict = {"lars": LARS, "sgd": SGD, "adam": Adam, "adamw": AdamW}

    def _get_optimizer(self, optimizer_type, lr, weight_decay=0, **opt_params):
        if optimizer_type not in self.__optimizer_dict:
            raise ValueError("Optimizer type needs to be in %r, got (%s)."
                             % (list(self.__optimizer_dict.keys()), optimizer_type))

        optimizer_class = self.__optimizer_dict[optimizer_type.lower()]

        if 'adam' in optimizer_type and 'momentum' in opt_params:
            # ignore momentum without raising error
            momentum = opt_params.pop('momentum')
            log.warning("[bold purple]Optimizer[/bold purple] %s optimizer doesn't use momentum. Momentum %.2e will "
                        "be ignored." % (optimizer_class.__name__, momentum), extra={"markup": True})
        return partial(optimizer_class, lr=lr, weight_decay=weight_decay, **opt_params)


class Trainer(ABC):
    r"""GPU Setup and Logging setup plus abstract funcs.
    """
    def __init__(self,
                 *,
                 distributed=False,
                 no_cuda=False,
                 gpu=None,
                 world_size=None,
                 rank=None,
                 master_gpu=None,
                 port=None,
                 log_steps,
                 logdir,
                 ):
        self.you_only_log_once = you_only_log_once()

        # Setup CPU and GPU
        assert not(distributed and no_cuda), 'Both flags for CPU-training and Multi-GPU training were set to True.'
        self.no_cuda = no_cuda
        self.distributed = distributed

        if self.no_cuda:
            # use cpu
            log.info('[bold]CUDA[/bold] Using CPU Device for training.', extra={"markup": True})
            self.device = torch.device('cpu')
        elif not self.distributed:
            # make sure gpus are available and that at least the gpu id is given.
            log.debug('[bold]CUDA[/bold] Found %d CUDA Devices..' % torch.cuda.device_count(), extra={"markup": True})

            assert torch.cuda.is_available(), 'No cuda devices found.'
            assert gpu is not None, 'Please specify which gpu to use.'
            self.device = torch.device(f'cuda:{gpu}')
            torch.cuda.set_device(self.device)
            log.info('[bold]CUDA[/bold] Single GPU Training, %r was selected.' % self.device, extra={"markup": True})

        if self.distributed:
            # Multi-GPU training
            # verify that all parameters are given.
            assert world_size is not None and rank is not None and gpu is not None and master_gpu is not None and \
            port is not None, 'Multi-GPU not setup correctly.'

            # verify that there are enough gpus.
            assert torch.cuda.device_count() >= world_size, 'Not enough GPUs, requested %d gpus, but only %d are ' \
                                                            'visible.' % (world_size, torch.cuda.device_count())

            self.device = torch.device(f'cuda:{gpu}')
            torch.cuda.set_device(self.device)
            self.gpu = gpu
            self.world_size = world_size
            self.rank = rank
            self.master_gpu = master_gpu
            self.port = port

            if self.info_w_all_ranks_allowed:
                log.info('[bold]CUDA[/bold] Distributed training, World size %d, Master gpu %d, %r has rank %d.'
                         % (self.world_size, self.master_gpu, self.device, self.rank), extra={"markup": True})
        else:
            # Single GPU/CPU training
            self.gpu = gpu
            self.rank = 0
            self.master_gpu = 0
            self.world_size = 1
            self.port = None

        # tensorboard logging
        self.logdir = os.path.expanduser(logdir) if logdir is not None else None  # remove home
        self.log_steps = log_steps

        if self.rank == 0:
            self.writer = SummaryWriter(self.logdir)
        if self.info:
            log.info('[bold]Logging[/bold] Tensorboard log and checkpoint directory: [u]%s[/u].' % self.logdir,
                     extra={"markup": True})

        # profiler
        if False and self.rank == 0:
            self.profiler = torch.profiler.profile(schedule=torch.profiler.schedule(
                    wait=2,
                    warmup=2,
                    active=6,
                    repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self.writer.log_dir),
                with_stack=True,
                profile_memory=True,
            )
        else:
            self.profiler = nullcontext()

        self.step = 0
        self.epoch = 0

    def __getattr__(self, item):
        if not (item in ['debug', 'info', 'debug_w_all_ranks_allowed', 'info_w_all_ranks_allowed']):
            raise AttributeError
        if item.startswith('debug'):
            out = log.isEnabledFor(logging.DEBUG)
        elif item.startswith('info'):
            out = log.isEnabledFor(logging.INFO)

        if '_w_all_ranks_allowed' in item:
            return out
        else:
            return out and (not hasattr(self, 'rank') or self.rank == 0)

    def _distribute_model(self, model):
        # Distribute model if needed
        # warning: random seed needs to have been fixed so all models share the same seed! or might be wrong because all
        # weights from rank 0 gpu are just copied
        assert self.distributed
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = self.port
        # todo does this need to be run in each node or only in node with rank 0? same question for new_group
        dist.init_process_group(backend='nccl', init_method='env://', rank=self.rank, world_size=self.world_size)
        self.group = dist.new_group()

        if self.debug_w_all_ranks_allowed:
            log.debug('[bold]CUDA[/bold] Distributed training (rank:%d), group initialize, port %r.'
                      % (self.rank, self.port), extra={"markup": True})

        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  # todo does this work
        # todo what does find_unused_parameters exactly do
        # is ema effected??
        # what about when the network gets to the mining phase, are parameters unused
        # change find_unused_parameters to False
        model = DDP(model, device_ids=[self.gpu], find_unused_parameters=True)
        if self.debug_w_all_ranks_allowed:
            log.debug('[bold]CUDA[/bold] Distributed training (rank:%d), model was wraped in DDP.' % self.rank,
                      extra={"markup": True})
        return model

    def _collect_params(self, modules):
        param_list = []
        for module in modules:
            for name, param in module.named_parameters():
                param_dict = {'params': param}
                param_list.append(param_dict)
        return param_list

    def _backward(self, loss, last=True):
        assert hasattr(self, 'no_sync_context_manager')
        if not last:
            with self.no_sync_context_manager():
                # accumulate gradients without syncing gpus
                loss.backward()
        else:
            loss.backward()

    def cleanup(self):
        if self.distributed:
            dist.destroy_process_group()
        if self.rank == 0:
            self.writer.close()
        if self.debug:
            log.debug('Cleanup done.')

    def log_scalars(self, loss):
        self.writer.add_scalar('train/loss', loss, self.step)
