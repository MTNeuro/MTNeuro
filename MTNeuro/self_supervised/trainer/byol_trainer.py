import hashlib
import os
from collections import OrderedDict
import re
from contextlib import nullcontext
import logging

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from MTNeuro.self_supervised.utils.console import console
from MTNeuro.self_supervised.models import BYOL, MLP3
from MTNeuro.self_supervised.loss import CosineLoss
from MTNeuro.self_supervised.utils.tensorboard import NamedTupleWrapper
from MTNeuro.self_supervised.utils import pass_on_failure
from MTNeuro.self_supervised.trainer import Trainer, ScheduleExtension, OptimizerSelectExtension


log = logging.getLogger(__name__)


class BYOLTrainer(Trainer, ScheduleExtension, OptimizerSelectExtension):
    r"""Trainer class for BYOL.

    Args:
        encoder,
        representation_size,
        projector_output_size,
        projector_hidden_size,
        symmetric_loss,
        train_dataloader,
        transform=None,
        transform_1=None,
        transform_2=None,
        prepare_views,
        total_epochs,
        batch_size,
        lr_warmup_epochs,
        base_lr,
        base_momentum,
        lr_decay,
        lr_poly_decay_n=None,
        mm_decay,
        optimizer_type,
        optimizer_momentum=None,
        weight_decay,
        exclude_bias_and_bn,
        distributed (bool, Optional): Flag for multi-GPU training (default: :obj:`False`)
        no_cuda (bool, Optional): Flag for CPU training. (default: :obj:`False`)
        gpu=None,
        world_size=None,
        rank=None,
        master_gpu=None,
        port=None,
        log_steps,
        logdir,
        resume_ckpt=None,

    ..note ::
    This class does not accept positional arguments or does not use unlimited keyword arguments to avoid default values
     being used without warning if the wrong keyword is used. All default values were removed. More sanity checks are
     performed inside during initialization.
    """
    def __init__(self,
                 *,
                 encoder,
                 representation_size,
                 projector=None,
                 predictor=None,
                 projector_output_size=None,
                 projector_hidden_size=None,
                 different_init=False,
                 symmetric_loss=True,
                 train_dataloader,
                 transform=None,
                 transform_1=None,
                 transform_2=None,
                 prepare_views,
                 total_epochs,
                 batch_size,
                 lr_warmup_epochs,
                 base_lr,
                 base_momentum,
                 use_lars_rule=True,
                 lr_decay,
                 lr_poly_decay_n=None,
                 lr_milestones=None,
                 lr_gamma=None,
                 mm_decay,
                 optimizer_type,
                 optimizer_momentum=None,
                 weight_decay,
                 exclude_bias_and_bn,
                 distributed=False,
                 no_cuda=False,
                 gpu=None,
                 world_size=None,
                 rank=None,
                 master_gpu=None,
                 port=None,
                 log_steps,
                 logdir,
                 log_img=False,
                 log_img_steps=None,
                 unnormalize=None,
                 resume_ckpt=None,
                 ):
        ################################
        # HANDLE GPU SETUP AND LOGGERS #
        ################################
        super().__init__(distributed=distributed, no_cuda=no_cuda, gpu=gpu, world_size=world_size,
                         rank=rank, master_gpu=master_gpu, port=port, log_steps=log_steps, logdir=logdir)

        ###############
        # DATALOADERS #
        ###############
        self.train_dataloader = train_dataloader
        self.prepare_views = prepare_views
        if isinstance(self.train_dataloader, list):
            self.num_examples = len(self.train_dataloader)
            log.info('[bold green]Dataset[/bold green] Got a list of samples.',
                     extra={"markup": True}) if self.info else None
        elif hasattr(self.train_dataloader, 'num_examples'):
            self.num_examples = self.train_dataloader.num_examples
        else: # elif graph dataloader, for now it's a single graph
            self.num_examples = len(self.train_dataloader.dataset)
        log.info('[bold green]Dataset[/bold green] Number of examples %d.' % self.num_examples,
                 extra={"markup": True}) if self.info else None

        ##############################
        # CLASSES OF TRANSFORMATIONS #
        ##############################
        # these are on gpu transforms! cpu transforms can be integrated inside of dataloaders
        # if transform is given then, the same class of transformation is used for both augmented views.
        assert not((transform is not None) and
                   (transform_1 is not None or transform_2 is not None)), 'Confusing, you can either specify a single' \
                                                                          'transform for generating both views OR one' \
                                                                          'class of transform for each. Got both.'
        assert (transform_1 is None) == (transform_2 is None), 'Only one class of transformation was specified.'

        self.transform_1 = transform_1 if transform_1 is not None else transform
        self.transform_2 = transform_2 if transform_2 is not None else transform

        if self.debug:
            log.debug('[bold green]Transformation class 1[/bold green] %r' % self.transform_1, extra={"markup": True})
            log.debug('[bold green]Transformation class 2[/bold green] %r' % self.transform_2, extra={"markup": True})

        ############################
        # TRAINING HYPERPARAMETERS #
        ############################
        # batch size
        self.train_batch_size = batch_size
        self.global_batch_size = self.world_size * self.train_batch_size
        log.info('[bold red]Hyperparameters[/bold red] Global batch size is %d.' % self.global_batch_size,
                 extra={"markup": True}) if self.info else None

        # number of epochs
        self.total_epochs = total_epochs
        self.total_steps = self.total_epochs * (self.num_examples // self.global_batch_size)

        log.info('[bold blue]Scheduler[/bold blue] The trainer will run %d epochs or %d training iterations '
                 'in total.' % (self.total_epochs, self.total_steps), extra={"markup": True}) if self.info else None

        # number of warmup epochs
        self.lr_warmup_epochs = lr_warmup_epochs
        self.warmup_steps = self.lr_warmup_epochs * (self.num_examples // self.global_batch_size)
        log.info('[bold blue]Scheduler[/bold blue] The learning rate decay will start after %d epochs or %d iterations.'
                 % (self.lr_warmup_epochs, self.warmup_steps), extra={"markup": True}) if self.info else None

        # learning rate used for online network update
        if use_lars_rule:
            # apply LARS learning rate scaling rule
            self.max_lr = (base_lr / 256) * self.global_batch_size
            log.info('[bold red]Hyperparameters[/bold red] The learning rate was updated according to the'
                     ' [u]LARS rule[/u], now %.2e.' % self.max_lr, extra={"markup": True}) if self.info else None
        else:
            self.max_lr = base_lr
            log.info('[bold red]Hyperparameters[/bold red] The learning rate is %.2e.' % self.max_lr,
                     extra={"markup": True}) if self.info else None

        # momentum used for target network EMA update
        self.base_mm = base_momentum
        log.info('[bold red]Hyperparameters[/bold red] The EMA momentum is %.2e.' % self.base_mm,
                 extra={"markup": True}) if self.info else None

        ##############
        # SCHEDULING #
        ##############
        # learning rate linear warmup then decay
        self.lr_decay = lr_decay
        self.lr_poly_decay_n = lr_poly_decay_n

        self.lr_milestones = [lr_m * (self.num_examples // self.global_batch_size) for lr_m in lr_milestones]  if \
            lr_milestones is not None else None
        self.lr_gamma = lr_gamma

        if self.info and self.lr_decay == 'cosine':
            log.info('[bold blue]Scheduler[/bold blue] The learning rate will follow a cosine decay.',
                      extra={"markup": True})
        elif self.info and self.lr_decay == 'poly':
            assert self.lr_poly_decay_n is not None
            log.info('[bold blue]Scheduler[/bold blue] The learning rate will follow a polynomial decay '
                     'with n=%f.' % self.lr_poly_decay_n, extra={"markup": True})
        elif self.info and self.lr_decay == 'step':
            assert self.lr_milestones is not None and self.lr_gamma is not None
            log.info('[bold blue]Scheduler[/bold blue] The learning rate will follow a step decay '
                     'with milestones=%s.' % self.lr_milestones, extra={"markup": True})

        # momentum decay
        self.mm_decay = mm_decay
        if self.info and self.mm_decay == 'cosine':
            log.info('[bold blue]Scheduler[/bold blue] The EMA momentum will follow a cosine decay '
                     'from %f to 1.' % self.base_mm, extra={"markup": True})
        elif self.info and self.mm_decay == 'cste':
            log.info('[bold blue]Scheduler[/bold blue] The EMA momentum will be fixed to %f.' % self.base_mm,
                     extra={"markup": True})

        ###############
        # BUILD MODEL #
        ###############
        with console.status("[bold green]Building model...") as status:
            self.representation_size = representation_size
            log.info('[bold red]Model[/bold red] Encoder %r with representation size %d.' % (encoder.__class__.__name__,
                     self.representation_size), extra={"markup": True}) if self.info else None

            # projector and predictor
            assert (projector is not None and predictor is not None) != \
                   (projector_output_size is not None and projector_hidden_size is not None)

            if projector is None:
                assert predictor is None  # should be None as well
                # using default projector/predictor
                log.info('[bold red]Model[/bold red] Using default projector/predictor architecture.',
                         extra={"markup": True}) if self.info else None
                projector, predictor = self._get_default_projector_predictor(representation_size, projector_output_size,
                                                                             projector_hidden_size)

            if self.debug:
                log.debug('[bold red]Model[/bold red] Projector %r' % projector, extra={"markup": True})
                log.debug('[bold red]Model[/bold red] Predictor %r' % predictor, extra={"markup": True})

            # build BYOL
            self.different_init = different_init
            log.info('[bold red]Model[/bold red] Online and target will%s be identically initialized.'
                     % (' not' if different_init else ''), extra={"markup": True}) if self.info else None
            self.symmetric_loss = symmetric_loss
            if self.symmetric_loss:
                log.info('[bold red]Model[/bold red] Trainer will use a symetrized loss.',
                         extra={"markup": True}) if self.info else None

            self.model = self._build_model(encoder, projector, predictor)
            log.info('[bold red]Model[/bold red] (rank:%d) Model built and lives on %r.'
                     % (self.rank, next(self.model.parameters()).device),
                     extra={"markup": True}) if self.info_w_all_ranks_allowed else None
            log.debug('[bold red]Model[/bold red] Architecture\n %r.' % self.model,
                      extra={"markup": True}) if self.debug else None

            ##################################
            # RESUME TRAINING: MODEL WEIGHTS #
            ##################################
            self.epoch = 0
            self.step = 0

            if resume_ckpt is not None:
                # load checkpoint
                log.info('[bold cyan]Checkpoint[/bold cyan] [blink]Loading model from %s.[/blink]' % resume_ckpt,
                         extra={"markup": True}) if self.info else None
                self._tmp_checkpoint = torch.load(resume_ckpt, map_location=self.device)

                # get epoch and step
                self.epoch = self._tmp_checkpoint['epoch']
                self.step = self._tmp_checkpoint['step']
                log.info('[bold cyan]Checkpoint[/bold cyan] Loaded model was trained for %d epochs or %d steps.'
                         % (self.epoch, self.step), extra={"markup": True}) if self.info else None

                # load model state dict
                self.load_model_from_checkpoint(self._tmp_checkpoint)

            ##############################################
            # MODEL IS BUILD AND WEIGHTS ARE INITIALIZED #
            # BROADCAST MODEL TO ALL GPU NODES           #
            ##############################################
            self.no_sync_context_manager = nullcontext
            if self.distributed:
                self.model = self._distribute_model(self.model)
                self.no_sync_context_manager = self.model.no_sync
                log.info('[bold]CUDA[/bold] Model was distributed.', extra={"markup": True}) if self.info else None

        #############
        # OPTIMIZER #
        #############
        with console.status("[bold green]Initializing optimizer...") as status:
            # collect params
            self.exclude_bias_and_bn = exclude_bias_and_bn
            log.info('[bold purple]Optimizer[/bold purple] Will exclude bias and batchnorm from weight decay.',
                     extra={"markup": True}) if self.info else None

            model_core = self.model if not self.distributed else self.model.module  # todo model.module causes inconsistencies?
            trainable_modules = [model_core.online_encoder, model_core.online_projector, model_core.predictor]
            params = self._collect_params(trainable_modules)
            log.debug('[bold purple]Optimizer[/bold purple] Collected parameters.',
                      extra={"markup": True}) if self.debug else None

            # define optimizer
            self.optimizer_momentum = optimizer_momentum
            self.weight_decay = weight_decay
            optimizer = self._get_optimizer(optimizer_type, lr=self.max_lr, momentum=self.optimizer_momentum,
                                            weight_decay=self.weight_decay)
            self.optimizer = optimizer(params)
            log.info('[bold purple]Optimizer[/bold purple] %s.' % self.optimizer.__class__.__name__,
                     extra={"markup": True}) if self.info else None

            ##############################
            # RESUME TRAINING: OPTIMIZER #
            ##############################
            if resume_ckpt is not None:
                # load optimizer
                self.load_optimizer_from_checkpoint(self._tmp_checkpoint)
                log.info('[bold cyan]Checkpoint[/bold cyan] Loaded optimizer state.',
                         extra={"markup": True}) if self.info else None

                # checksum test
                if self._tmp_checkpoint['checksum'] != self.checksum:
                    log.error('[bold cyan]Checkpoint[/bold cyan] Checksum test failed. Hyperparameters do not match.',
                              extra={"markup": True}) if self.info else None
                else:
                    log.info('[bold cyan]Checkpoint[/bold cyan] Checksum match.',
                             extra={"markup": True}) if self.info else None

        #################
        # LOSS FUNCTION #
        #################
        log.info('[bold red]Model[/bold red] Using cosine loss.',
                 extra={"markup": True}) if self.info else None
        self.criterion = CosineLoss()

        #############################
        # LOGGING AND CHECKPOINTING #
        #############################
        # logging images
        assert not(log_img and (log_img_steps is None or unnormalize is None)), \
            'Please specify log_img_steps/unormalize.'
        self.log_img = log_img
        self.log_img_steps = log_img_steps
        self.unnormalize = unnormalize

        # checkpoint pattern
        self.ckpt_path_format = os.path.join(self.logdir, "ckpt-%d.pt") if self.logdir is not None else None

    def _get_default_projector_predictor(self, input_size, projector_output_size, projector_hidden_size):
        projector = MLP3(input_size, projector_output_size, projector_hidden_size)
        predictor = MLP3(projector_output_size, projector_output_size, projector_hidden_size)
        return projector, predictor

    def _build_model(self, encoder, projector, predictor):
        # todo should everything be copied to gpu first? saves the transfer for target nets
        # does deepcopy work on gpus?
        # todo add assert different init to check that all modules have reset_parameters method
        net = BYOL(encoder, projector, predictor, different_init=self.different_init)
        return net.to(self.device)

    def load_model_from_checkpoint(self, checkpoint):
        self.model.load_state_dict(checkpoint['model'], strict=True)
        return True

    def load_optimizer_from_checkpoint(self, checkpoint):
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return True

    def _collect_params(self, modules):
        """
        exclude_bias_and bn: exclude bias and bn from both weight decay and LARS adaptation
            in the PyTorch implementation of ResNet, `downsample.1` are bn layers
        """
        param_list = []
        for module in modules:
            for name, param in module.named_parameters():
                if self.exclude_bias_and_bn and ('bn' in name or 'downsample.1' in name or 'bias' in name):
                    param_dict = {'params': param, 'weight_decay': 0., 'lars_exclude': True}
                else:
                    param_dict = {'params': param}
                param_list.append(param_dict)
        return param_list

    @property
    @pass_on_failure  # if fails returns None # todo rename to safemethod # todo only once
    def checksum(self):
        attrs = [self.representation_size, self.projector_output_size, self.projector_hidden_size, self.symmetric_loss,
                 self.total_epochs, self.global_batch_size, self.lr_warmup_epochs, self.max_lr, self.base_mm,
                 self.lr_decay, self.lr_poly_decay_n, self.mm_decay, self.optimizer, self.optimizer_momentum,
                 self.weight_decay, self.exclude_bias_and_bn]
        attrs_str = '-'.join([a.__repr__() for a in attrs])
        return hashlib.md5(attrs_str.encode('utf-8')).hexdigest()

    def update_learning_rate(self, step):
        lr = self._warmup_then_decay(step, max_val=self.max_lr, warmup_steps=self.warmup_steps,
                                     total_steps=self.total_steps, warmup_type='linear', decay_type=self.lr_decay,
                                     decay_n=self.lr_poly_decay_n, decay_milestones=self.lr_milestones,
                                     decay_gamma=self.lr_gamma)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def update_momentum(self, step):
        if self.mm_decay == 'cste':
            self.mm = self.base_mm
        else:
            self.mm = 1 - self._decay(step, max_val=(1 - self.base_mm), total_steps=self.total_steps,
                                      decay_type=self.mm_decay)

    def save_checkpoint(self, epoch):
        if self.rank == 0:
            path = self.ckpt_path_format % epoch
            state = {
                'optimizer': self.optimizer.state_dict(),
                'epoch': self.epoch,
                'step': self.step,
                'checksum': self.checksum
            }
            # recommended way of saving checkpoint from distributed model
            if not self.distributed:
                state['model'] = self.model.state_dict()
            else:
                state['model'] = self.model.module.state_dict()
            torch.save(state, path)
        if self.distributed:
            # create barrier to lock all nodes
            # todo will probably wait either way in the forward. rm
            dist.barrier()

    def update_target_network(self):
        if not self.distributed:
            self.model.update_target_network(self.mm)
        else:
            # todo modifying weights should be ok but needs to be verified
            self.model.module.update_target_network(self.mm)

    @staticmethod
    def load_trained_encoder(encoder, ckpt_path, device):
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint['model']

        # extract online encoder
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            if 'online_encoder' in key:
                # remove online_encoder
                new_key = re.sub(r'^online_encoder\.', '', key)
                new_state_dict[new_key] = value
        encoder.load_state_dict(new_state_dict, strict=True)
        return checkpoint['epoch']

    def log_scalars(self, loss):
        self.writer.add_scalar('params/lr', self.optimizer.param_groups[0]['lr'], self.step)
        self.writer.add_scalar('params/mm', self.mm, self.step)
        self.writer.add_scalar('train/loss', loss, self.step)

    @pass_on_failure
    def log_views(self, view1, view2, tag):
        r"""
        ..note ::
        Only works with a multiple of 16
        """
        assert view1.ndim in [3, 4]
        img_batch = np.zeros((16, *view1.shape[1:]))
        view1, view2 = view1[:16].cpu(), view2[:16].cpu() # todo does this only copy the slice
        for i in range(8):
            img_batch[i] = self.unnormalize(view1[i]).numpy()
            img_batch[8+i] = self.unnormalize(view2[i]).numpy()

        # to avoid color distortion in tensorboard
        # todo still doesn't work
        img_batch = np.clip(255 * img_batch, 0, 255).astype('uint8')
        self.writer.add_images(tag, img_batch, self.step)

    def _process_inputs(self, inputs):
        inputs = self.prepare_views(inputs)
        out = {}

        for key, required in [('view_1', True), ('view_2', True)]:
            if required or key in inputs:
                if isinstance(inputs[key], str):
                    # avoids copying to gpu twice
                    copy_from = inputs[key]
                    out[key] = out[copy_from]  # todo add deep copy in graphs
                else:
                    # todo skip when no mining will be done
                    out[key] = inputs[key].to(self.device, non_blocking=True)
                del inputs[key]
        # todo add flags to check when using transform is needed
        return out

    def _input_repr(self, data):
        if isinstance(data, torch.Tensor):
            return 'shape {}'.format(data.size())
        else:
            # todo add graph instance check
            return data.__repr__()

    def train_epoch(self):
        self.model.train()

        for inputs in self.train_dataloader:
            # update parameters
            self.update_learning_rate(self.step)
            self.update_momentum(self.step)

            # get inputs
            _, labels = inputs
            inputs = self._process_inputs(inputs)
            view_1 = inputs['view_1']
            view_2 = inputs['view_2']
            

            with self.you_only_log_once(self.debug) as go_ahead:
                log.debug('[bold]Input sample[/bold] %r' % self._input_repr(view_1),
                          extra={"markup": True}) if go_ahead else None


            # apply transforms
            view_1, _ = self.transform_1(view_1, label = labels.to(self.device)) if self.transform_1 else view_1
            view_2, _ = self.transform_2(view_2, label = labels.to(self.device)) if self.transform_1 else view_2

            # debugging
            with self.you_only_log_once(self.debug) as go_ahead:
                log.debug('[bold]Augmented View[/bold] %r' % self._input_repr(view_1),
                          extra={"markup": True}) if go_ahead else None

            with self.you_only_log_once(False) as go_ahead:
                # todo does not work for graphs
                # add graph to tensorboard
                self.writer.add_graph(NamedTupleWrapper(self.model),
                                      {'online_view': view_1, 'target_view': view_2}) if go_ahead else None

            # forward
            self.optimizer.zero_grad()
            outputs = self.model({'online_view': view_1, 'target_view': view_2})

            weight = 1. if not self.symmetric_loss else 0.5
            if hasattr(view_1, 'node_mask'):
                outputs['online_q'] = outputs['online_q'][view_1.node_mask]
            if hasattr(view_2, 'node_mask'):
                outputs['target_z'] = outputs['target_z'][view_2.node_mask]
            loss1 = weight * self.criterion(outputs['online_q'], outputs['target_z'])
            self._backward(loss1, last=not self.symmetric_loss)

            if self.symmetric_loss:
                outputs = self.model({'online_view': view_2, 'target_view': view_1})
                if hasattr(view_2, 'node_mask'):
                    outputs['online_q'] = outputs['online_q'][view_2.node_mask]
                if hasattr(view_1, 'node_mask'):
                    outputs['target_z'] = outputs['target_z'][view_1.node_mask]
                loss2 = 0.5 * self.criterion(outputs['online_q'], outputs['target_z'])
                self._backward(loss2, last=True)

            # update online network
            self.optimizer.step()

            if False and self.profiler:
                # todo fix profiler
                self.profiler.step()

            # update moving average
            # todo is removing grad enough? what happens to batchnorm, does it need to be in eval mode??
            self.update_target_network()

            # log scalars
            if self.rank == 0 and self.step % self.log_steps == 0:
                loss = loss1.item() + loss2.item() if self.symmetric_loss else loss1.item()
                self.log_scalars(loss=loss)

            # log images
            if self.rank == 0 and self.log_img and self.step % self.log_img_steps:
                self.log_views(view_1, view_2, 'Augmented Views')

            # update step
            self.step += 1
        self.epoch += 1
