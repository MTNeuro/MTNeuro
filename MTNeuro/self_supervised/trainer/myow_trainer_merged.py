import logging

import torch
import torch.distributed as dist
import sklearn
try:
    import torch_geometric
except ImportError:
    torch_geometric = None

from MTNeuro.self_supervised.utils.console import console
from MTNeuro.self_supervised.models import MYOW
from MTNeuro.self_supervised.trainer import BYOLTrainer
from MTNeuro.self_supervised.utils.tensorboard import NamedTupleWrapper
from MTNeuro.self_supervised import visualization
from MTNeuro.self_supervised.utils import suppress_logs, pass_on_failure


log = logging.getLogger(__name__)


class MYOWTrainerMerged(BYOLTrainer):
    r"""Trainer class for MYOW."""
    def __init__(self,
                 *,
                 encoder,
                 representation_size,
                 projector=None,
                 predictor=None,
                 projector_output_size=None,
                 projector_hidden_size=None,
                 different_init=False,
                 symmetric_loss,
                 myow_only=False,
                 layout='cascaded',
                 projector_m=None,
                 predictor_m=None,
                 projector_2_output_size=None,
                 projector_2_hidden_size=None,
                 train_dataloader,
                 transform=None,
                 transform_1=None,
                 transform_2=None,
                 view_pool_dataloader=None,
                 transform_m=None,
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
                 byol_warmup_epochs,  # number of epochs where only byol is running
                 myow_rampup_epochs,  # number of epochs where myow's weights is ramped up
                 base_myow_weight,
                 view_miner,  #='knn',
                 view_miner_candidate_repr, #='target'
                 view_miner_distance, #='cosine',
                 select_neigh=None, #='random',
                 knn_nneighs=None,
                 mining_threshold=None,
                 scale_threshold=False,
                 gamma=None, #=1e0,
                 niters=None, #=10,
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
                 convert_byol_to_myow=False,
                 ):

        with console.status("[bold green]Building mining block...") as status:
            #########################
            # VIEW POOL DATALOADERS #
            #########################
            self.view_pool_dataloader = view_pool_dataloader
            if self.view_pool_dataloader is not None:
                log.debug('[bold red]Dataset[/bold red] A view dataloader will be used, for mining',
                          extra={"markup": True}) if self.debug else None

            ##########################
            # MINING TRANSFORMATIONS #
            ##########################
            self.transform_m = transform_m
            log.debug('[bold green]Transformation class m[/bold green] %r' % self.transform_m,
                      extra={"markup": True}) if self.debug else None
            # todo else: verify that prepare views is working as expected: give 4 views.
            # if view pool dataloader, assume that prepare views works for both

            ############################
            # TRAINING HYPERPARAMETERS #
            ############################
            if isinstance(train_dataloader, list):
                self.num_examples = len(train_dataloader)
            elif hasattr(train_dataloader, 'num_examples'):
                self.num_examples = train_dataloader.num_examples
            else:
                self.num_examples = len(train_dataloader.dataset)

            self.world_size = world_size if distributed else 1

            self.train_batch_size = batch_size
            self.global_batch_size = self.world_size * self.train_batch_size

            self.byol_warmup_epochs = byol_warmup_epochs
            self.byol_warmup_steps = self.byol_warmup_epochs * (self.num_examples // self.global_batch_size)
            log.info('[bold blue]Scheduler[/bold blue] BYOL will run solo for %d epochs or %d training iterations, '
                     'before the MYOW loss is introduced.' % (self.byol_warmup_epochs, self.byol_warmup_steps),
                     extra={"markup": True}) if self.info else None

            self.base_myow_weight = base_myow_weight
            self.myow_rampup_epochs = myow_rampup_epochs
            self.myow_rampup_steps = self.myow_rampup_epochs * (self.num_examples // self.global_batch_size) # todo add no rampup
            log.info('[bold blue]Scheduler[/bold blue] MYOW will rampup from 0. to %.2f in %d epochs or %d training '
                     'iterations.' % (self.base_myow_weight, self.myow_rampup_epochs, self.myow_rampup_steps),
                     extra={"markup": True}) if self.info else None

            ######################
            # ADDITIONAL MODULES #
            ######################
            self.layout = layout
            log.info('[bold red]Model[/bold red] Projector layout: %s.' % self.layout,
                     extra={"markup": True}) if self.info else None

            # second projector and predictor
            assert (projector_m is not None and predictor_m is not None) != \
                   (projector_2_output_size is not None and projector_2_hidden_size is not None)

            if projector_m is None:
                assert predictor_m is None  # should be None as well
                # using default projector/predictor
                log.info('[bold red]Model[/bold red] Using default projector_m/predictor_m architecture.',
                         extra={"markup": True}) if self.info else None
                # todo change 2 to m
                input_size = {'parallel': representation_size, 'cascaded': projector_output_size}[self.layout]
                if projector_2_output_size != 0:
                    projector_m, predictor_m = self._get_default_projector_predictor(input_size,
                                                                                     projector_2_output_size,
                                                                                     projector_2_hidden_size)
                else:
                    projector_m = torch.nn.Identity()
                    _, predictor_m = self._get_default_projector_predictor(input_size,
                                                                           projector_output_size,
                                                                           projector_hidden_size)

            log.debug('[bold red]Model[/bold red] Projector_m %r' % projector_m,
                      extra={"markup": True}) if self.debug else None

            self.__additional_modules = {'projector_m': projector_m, 'predictor_m': predictor_m}

            ###############
            # VIEW MINING #
            ###############
            assert view_miner_candidate_repr in ['online', 'target']
            self.view_miner_candidate_repr = view_miner_candidate_repr
            log.info('[bold pink]Mining[/bold pink] Mining will be done in: %s.' % self.view_miner_candidate_repr,
                     extra={"markup": True}) if self.info else None
            assert view_miner in ['knn', 'ot'], 'View miner needs to be either knn or ot, got %s.' % view_miner
            self.view_miner = view_miner
            assert view_miner_distance in ['cosine'], 'Only cosine distance currently implemented.'
            self.view_miner_distance = view_miner_distance
            log.info('[bold pink]Mining[/bold pink] %s will be the mined and will use a %s distance.'
                     % (self.view_miner, self.view_miner_distance), extra={"markup": True}) if self.info else None

            if self.view_miner == 'knn':
                assert select_neigh is not None and knn_nneighs is not None, 'parameters missing.'
                log.info('[bold pink]Mining[/bold pink] One of %d-nearset neighbors will be selected as the mined view.'
                         %knn_nneighs, extra={"markup": True}) if self.info else None
            elif self.view_miner == 'ot':
                assert gamma is not None and niters is not None, 'parameters missing.'

            # knn view miner
            self.select_neigh = select_neigh
            self.knn_nneighs = knn_nneighs

            # ot view miner
            self.gamma = gamma
            self.niters = niters

            # threshold
            self.mining_threshold = mining_threshold if mining_threshold is not None else torch.finfo(torch.float32).max
            self.scale_threshold = scale_threshold

            self.myow_only = myow_only
        ###################
        # RESUME TRAINING #
        ###################
        self.convert_byol_to_myow = convert_byol_to_myow
        if self.convert_byol_to_myow:
            log.info('[bold pink]Checkpoint[/bold pink] Will be converting a BYOL checkpoint to MYOW, which means that '
                     'new modules will be initialized randomly.', extra={"markup": True}) if self.info else None
        self.__optimizer_complete = False

        super().__init__(encoder=encoder,
                         representation_size=representation_size,
                         projector=projector,
                         predictor=predictor,
                         projector_output_size=projector_output_size,
                         projector_hidden_size=projector_hidden_size,
                         different_init=different_init,
                         symmetric_loss=symmetric_loss,
                         train_dataloader=train_dataloader,
                         transform=transform,
                         transform_1=transform_1,
                         transform_2=transform_2,
                         prepare_views=prepare_views,
                         total_epochs=total_epochs,
                         batch_size=batch_size,
                         lr_warmup_epochs=lr_warmup_epochs,
                         base_lr=base_lr,
                         base_momentum=base_momentum,
                         use_lars_rule=use_lars_rule,
                         lr_decay=lr_decay,
                         lr_poly_decay_n=lr_poly_decay_n,
                         lr_milestones=lr_milestones,
                         lr_gamma=lr_gamma,
                         mm_decay=mm_decay,
                         optimizer_type=optimizer_type,
                         optimizer_momentum=optimizer_momentum,
                         weight_decay=weight_decay,
                         exclude_bias_and_bn=exclude_bias_and_bn,
                         distributed=distributed,
                         no_cuda=no_cuda,
                         gpu=gpu,
                         world_size=world_size,
                         rank=rank,
                         master_gpu=master_gpu,
                         port=port,
                         log_steps=log_steps,
                         logdir=logdir,
                         log_img=log_img,
                         log_img_steps=log_img_steps,
                         unnormalize=unnormalize,
                         resume_ckpt=resume_ckpt)

        ################################
        # ADD NEW MODULES TO OPTIMIZER #
        ################################
        with console.status("[bold green]Finish building optimizer...") as status:
            model_core = self.model if not self.distributed else self.model.module
            additional_trainable_modules = [model_core.online_projector_m, model_core.predictor_m]
            additional_params = self._collect_params(additional_trainable_modules)

            for param_group in additional_params:
                # add groups one by one
                self.optimizer.add_param_group(param_group)

            log.debug('[bold purple]Optimizer[/bold purple] Collected parameters from additional modules.',
                      extra={"markup": True}) if self.debug else None

            ################################
            # RESUME TRAINING: OPTIMIZER   #
            ################################
            if not self.convert_byol_to_myow and resume_ckpt is not None:
                self.__optimizer_complete = True
                self.load_optimizer_from_checkpoint(self._tmp_checkpoint)
                log.debug('[bold purple]Optimizer[/bold purple] Loaded optimizer state from a MYOW checkpoint.',
                          extra={"markup": True}) if self.debug else None

    def _build_model(self, encoder, projector, predictor):
        # this func will be called in BYOLTrainer, so the additional modules are passed through self
        projector_m = self.__additional_modules['projector_m']
        predictor_m = self.__additional_modules['predictor_m']
        net = MYOW(encoder, projector, projector_m, predictor, predictor_m, n_neighbors=self.knn_nneighs,
                   distance=self.view_miner_distance, view_miner=self.view_miner, select_neigh=self.select_neigh,
                   gamma=self.gamma, niters=self.niters, different_init=self.different_init, layout=self.layout)
        del self.__additional_modules
        return net.to(self.device)

    def load_model_from_checkpoint(self, checkpoint):
        if self.convert_byol_to_myow:
            out = self.model.load_state_dict(checkpoint['model'], strict=False)
            # verify that the only difference is in the second projector/predictor networks.
            assert len(out.unexpected_keys) == 0, 'Got unexpected keys, %r.' % out.unexpected_keys
            assert all(['_m' in key for key in out.missing_keys]), 'Got missing keys, %r.' % out.missing_keys

            # verify that checkpoint was trained for the right number of epochs.
            assert checkpoint['epoch'] == self.byol_warmup_epochs, 'BYOL-only phase ends at %d but got checkpoint' \
                                                                   'trained for %d.' % (self.byol_warmup_epochs,
                                                                                        checkpoint['epoch'])
        else:
            # regular resuming
            self.model.load_state_dict(checkpoint['model'], strict=True)

    def load_optimizer_from_checkpoint(self, checkpoint):
        if self.convert_byol_to_myow or self.__optimizer_complete:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def update_mined_loss_weight(self, step):
        max_w = self.base_myow_weight
        min_w = 0.
        if step < self.byol_warmup_steps:
            self.mined_loss_weight = min_w
        elif step >= (self.byol_warmup_steps + self.myow_rampup_steps):
            self.mined_loss_weight = max_w
        else:
            self.mined_loss_weight = min_w + (max_w - min_w) * (step - self.byol_warmup_steps) / self.myow_rampup_steps

    def log_scalars(self, loss, loss_m):
        super().log_scalars(loss)
        self.writer.add_scalar('params/myow_weight', self.mined_loss_weight, self.step)
        if loss_m is not None:
            self.writer.add_scalar('train/loss_mine', loss_m, self.step)

    @pass_on_failure
    @suppress_logs('info')  # matplotlib debug msgs
    def log_miner_stats(self, correspondence, track_across_steps=False):  # todo add types
        # finds out how many hubs there are
        _, hubs = torch.unique(correspondence, return_counts=True)
        self.writer.add_histogram('miner/hubs', hubs, self.step, bins='auto')

        if track_across_steps:
            if not hasattr(self, '_prev_correspondence'):
                self._prev_correspondence = correspondence
            else:
                same = correspondence.eq(self._prev_correspondence.view_as(correspondence)).sum().item()
                same_percentage = same / len(correspondence)
                self.writer.add_scalar('miner/same_mined', same_percentage, self.step)
                self._prev_correspondence = correspondence

    @pass_on_failure
    @suppress_logs('info')  # matplotlib debug msgs
    def log_miner_dist(self, dist):  # todo add types
        # finds out how many hubs there are
        self.writer.add_histogram('miner/dist', dist, self.step, bins='auto')

    @pass_on_failure
    @suppress_logs('info')  # matplotlib debug msgs
    def log_miner_confusion_matrix(self, labels, labels_pool, correspondence, class_names=None, comment=''):
        cm = sklearn.metrics.confusion_matrix(labels, labels_pool[correspondence])
        figure = visualization.plot_confusion_matrix(cm, class_names=class_names,
                                                     xlabel='Label of mined view', ylabel='Label of sample')
        self.writer.add_figure('miner/confusion_matrix' + comment, figure, self.step, close=True)
        acc = cm.diagonal().sum() / cm.sum()
        self.writer.add_scalar('miner/acc' + comment, acc, self.step)

    def _next_in_pool_dataloader(self):
        try:
            # needs to be the same structure as data
            # todo use label_pool to get percentage of false positive views.
            view_pool, label_pool = next(self.view_pooler)
        except StopIteration:
            # reinit the dataloader
            if self.distributed:
                log.error('[red]In distributed mode, the same sequence of samples will be seen during mining.[/red]',
                          extra={"markup": True})
            self.view_pooler = iter(self.view_pool_dataloader)
            view_pool, label_pool = next(self.view_pooler)
        return view_pool, label_pool

    def _process_inputs(self, inputs):
        inputs = self.prepare_views(inputs)
        out = {}
        view_pool_required = True

        if self.view_pool_dataloader is not None:
            assert 'view_pool' not in inputs and 'metadata_pool' not in inputs
            # todo add warning that transform was not applied
            view_pool, view_pool_labels = self._next_in_pool_dataloader()
            out['view_pool'] = view_pool.to(self.device, non_blocking=True)
            out['metadata_pool'] = {'labels': view_pool_labels}
            view_pool_required = False

            assert 'view_m' not in inputs
            inputs['view_m'] = 'view_1'

        for key, required in [('view_1', True), ('view_2', True), ('view_m', True), ('view_pool', view_pool_required),
                              ('cand_edge_index', False), ('ccand_edge_index', False),
                              ('metadata_m', False), ('metadata_pool', False), ('metadata_header', False)]:
            if required or key in inputs:
                if isinstance(inputs[key], str):
                    # avoids copying to gpu twice
                    copy_from = inputs[key]
                    out[key] = out[copy_from]  # todo add deep copy in graphs
                elif isinstance(inputs[key], torch.Tensor) or \
                        (torch_geometric is not None and isinstance(inputs[key], torch_geometric.data.Data)):
                    # todo skip when no mining will be done
                    out[key] = inputs[key].to(self.device, non_blocking=True)
                else:
                    # metadata lists or dicts
                    out[key] = inputs[key]
                del inputs[key]
        return out

    def transform_for_mining(self, view_m, view_pool=None, mine_mask=None):
        with self.you_only_log_once(self.debug) as go_ahead:
            log.debug('[bold]View m[/bold] %r' % self._input_repr(view_m),
                      extra={"markup": True}) if go_ahead else None

        if self.transform_m is not None:
            if view_pool is None:
                # mine within the same batch
                # todo add default diagonal matrix for restriction
                with self.you_only_log_once(self.debug) as go_ahead:
                    log.debug('[bold]Self mining![/bold]', extra={"markup": True}) if go_ahead else None

                view_m, _ = self.transform_m(view_m, None)

                if False and torch_geometric is not None and isinstance(view_m, torch_geometric.data.Data) and mine_mask is not None:
                    edge_index = mine_mask._indices()
                    contains_isolated_nodes = \
                        torch_geometric.utils.contains_isolated_nodes(edge_index, num_nodes=view_m.num_nodes)

                    with self.you_only_log_once(self.debug) as go_ahead:
                        if go_ahead and contains_isolated_nodes:
                            log.debug('[bold]Graph mining[/bold] Isolated nodes will be removed.',
                                      extra={"markup": True})
                    if contains_isolated_nodes:
                        # remove isolated nodes
                        edge_index, _, mask = torch_geometric.utils.remove_isolated_nodes(edge_index,
                                                                                          num_nodes=view_m.num_nodes)
                        mine_mask = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1), dtype=torch.bool),
                                                            device=edge_index.device)
                        view_m.edge_index, _ = torch_geometric.utils.subgraph(mask, view_m.edge_index, relabel_nodes=True)
                        view_m.x, view_m.y = view_m.x[mask], view_m.y[mask]
                return view_m, None, mine_mask
            else:
                with self.you_only_log_once(self.debug) as go_ahead:
                    log.debug('[bold]View pool[/bold] %r' % self._input_repr(view_pool),
                              extra={"markup": True}) if go_ahead else None

                if torch_geometric is not None and isinstance(view_m, torch_geometric.data.Data):
                    # todo collate batch of batches
                    raise NotImplementedError('graph collate not implemented')

                # collate (in case the same transform is applied)
                views_to_aug = torch.cat([view_m, view_pool], dim=0)

                # transform
                views_aug, _ = self.transform_m(views_to_aug, None)

                # split back
                view_m = views_aug[:view_m.size(0)]
                view_pool = views_aug[view_m.size(0):]
                return view_m, view_pool, mine_mask
        else:
            # no transform
            return view_m, view_pool, mine_mask

    @torch.no_grad()
    def mine(self, view_m, online_y_m, view_pool, cand_edge_index=None, ccand_edge_index=None):
        # compute representations for candidates
        if self.view_miner_candidate_repr == 'online':
            # todo if online is better add symmetry and use all computations. Can even have a larger memory bank.
            if view_pool is None:
                # self mining! no need to recompute representations
                y_pool = online_y_m.detach()
            else:
                # feed candidate samples through online encoder
                outputs_pool = self.model({'online_view': view_pool}, get_embedding='encoder')
                y_pool = outputs_pool['online_y']

        elif self.view_miner_candidate_repr == 'target':
            if view_pool is None:
                # forward main samples through target encoder
                # the representation will be kept for later!
                outputs_pool = self.model({'target_view': view_m}, get_embedding='encoder')
                y_pool = outputs_pool['target_y']
            else:
                # feed candidate samples through target encoder
                outputs_pool = self.model({'target_view': view_pool}, get_embedding='encoder')
                y_pool = outputs_pool['target_y']

        # gather all candidates from all gpus
        if self.distributed:
            if cand_edge_index is not None or ccand_edge_index is not None:
                # todo it also needs to be broadcasted
                raise NotImplementedError
            gather_list = [torch.zeros_like(y_pool) for _ in range(self.world_size)]
            dist.all_gather(gather_list, y_pool, self.group)
            y_pool = torch.cat(gather_list, dim=0)
            with self.you_only_log_once(self.debug) as go_ahead:
                log.debug('[bold]Distributed gather pool batch[/bold] %r' % self._input_repr(y_pool),
                          extra={"markup": True}) if go_ahead else None

        # mine views
        if cand_edge_index is not None or ccand_edge_index is not None:
            with self.you_only_log_once(self.debug) as go_ahead:
                log.debug('[bold]Sparse Mining mask[/bold] %r' % self._input_repr(cand_edge_index),
                          extra={"markup": True}) if go_ahead else None
                log.debug('[bold]Dense Mining mask[/bold] %r' % self._input_repr(ccand_edge_index),
                          extra={"markup": True}) if go_ahead else None

        mine_views = self.model.mine_views if not self.distributed else self.model.module.mine_views
        selection_mask, mined_dist = mine_views(online_y_m, y_pool, cand_edge_index=cand_edge_index, ccand_edge_index=ccand_edge_index)
        return selection_mask, mined_dist, y_pool

    def train_epoch(self):
        self.model.train()

        if self.view_pool_dataloader is not None:
            # for images
            self.view_pooler = iter(self.view_pool_dataloader)

        for inputs in self.train_dataloader:
            # todo main batch in mining
            #####################
            # UPDATE PARAMETERS #
            #####################
            self.update_learning_rate(self.step)
            self.update_momentum(self.step)
            self.update_mined_loss_weight(self.step)

            ############################################
            # PROCESS INPUTS AND APPLY TRANSFORMATIONS #
            ############################################
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

            with self.you_only_log_once(self.debug) as go_ahead:
                log.debug('[bold]Augmented View[/bold] %r' % self._input_repr(view_1),
                          extra={"markup": True}) if go_ahead else None

            with self.you_only_log_once(False) as go_ahead:
                # add graph to tensorboard
                self.writer.add_graph(NamedTupleWrapper(self.model),
                                      {'online_view': view_1, 'target_view': view_2}) if go_ahead else None

            if not self.myow_only:
                #############
                # BYOL TERM #
                #############
                self.optimizer.zero_grad()
                outputs = self.model({'online_view': view_1, 'target_view': view_2})

                weight = 1. / (1. + self.mined_loss_weight)
                weight = weight * 0.5 if self.symmetric_loss else weight
                loss1 = weight * self.criterion(outputs['online_q'], outputs['target_z'])
                self._backward(loss1, last=(not self.symmetric_loss and self.mined_loss_weight == 0.))

                ########################
                # BYOL TERM: SYMMETRIC #
                ########################
                if self.symmetric_loss:
                    outputs = self.model({'online_view': view_2, 'target_view': view_1})
                    weight = 1 / (1. + self.mined_loss_weight) / 2.
                    loss2 = weight * self.criterion(outputs['online_q'], outputs['target_z'])
                    self._backward(loss2, last=self.mined_loss_weight == 0.)

            #############
            # MYOW TERM #
            #############
            if self.mined_loss_weight > 0:
                #############################
                # GET INPUTS AND MINE VIEWS #
                #############################
                # get pool of candidates
                view_m = inputs['view_m']
                view_pool = inputs['view_pool']
                cand_edge_index = inputs['cand_edge_index'] if 'cand_edge_index' in inputs else None
                ccand_edge_index = inputs['ccand_edge_index'] if 'ccand_edge_index' in inputs else None

                # apply transform
                view_m, view_pool, _ = self.transform_for_mining(view_m, view_pool, None)

                # compute representations for main batch
                outputs = self.model({'online_view': view_m}, get_embedding='encoder')
                online_y = outputs['online_y']

                # mine
                selection_mask, mined_dist, y_pool = self.mine(view_m, online_y, view_pool, cand_edge_index, ccand_edge_index)

                # threshold
                keep_mask = mined_dist <= self.mining_threshold

                #################
                # FORWARD VIEWS #
                #################
                # todo add option to forward aug 1 instead
                view_pool = view_m if view_pool is None else view_pool
                if self.view_miner_candidate_repr == 'online':
                    if self.distributed:
                        # need to broadcast the inputs...
                        # todo feed batch to online and target and broadcast. loses advantage when B < L
                        # todo only do it in distributed then
                        raise NotImplementedError()

                    # the views need to be forwarded through the target network
                    if torch_geometric is None or not isinstance(view_pool, torch_geometric.data.Data):
                        # can save computation by only forwarding the selected samples
                        target_view_mined = view_pool[selection_mask][keep_mask]
                        outputs_mined = self.model({'online_view': online_y[keep_mask], 'target_view': target_view_mined})
                    else:
                        # all of the graph needs to be forwarded
                        outputs_pool = self.model({'target_view': view_pool}, get_embedding='encoder')
                        # now select
                        target_y_mined = outputs_pool['target_y'][selection_mask][keep_mask]
                        outputs_mined = self.model({'online_view': online_y[keep_mask], 'target_view': target_y_mined})

                elif self.view_miner_candidate_repr == 'target':
                    # already computed, just forward
                    target_y_mined = view_pool[selection_mask][keep_mask]
                    outputs_mined = self.model({'online_view': view_m[keep_mask], 'target_view': target_y_mined})

                # backward
                weight = self.mined_loss_weight / (1. + self.mined_loss_weight)
                if self.scale_threshold:
                    weight *= keep_mask.sum() / keep_mask.size(0)
                weight = weight * 0.5 if self.symmetric_loss else weight
                loss3 = weight * self.criterion(outputs_mined['online_q'], outputs_mined['target_z'])
                self._backward(loss3, last= not self.symmetric_loss)
                
                if self.symmetric_loss:
                    outputs_mined = self.model({'online_view': target_y_mined, 'target_view': view_m[keep_mask]})
                    loss4 = weight * self.criterion(outputs_mined['online_q'], outputs_mined['target_z'])
                    self._backward(loss4, last= True)

            # update online network
            self.optimizer.step()

            # update moving average
            self.update_target_network()

            # log
            if self.rank == 0 and self.step % self.log_steps == 0:
                loss = loss1.item() + loss2.item() if self.symmetric_loss else loss1.item() if not self.myow_only else 0.
                loss *= (1. + self.mined_loss_weight)  # rescale so it's between 0 and 1

                loss_m = loss3.item() if self.mined_loss_weight > 0 else 0.
                loss_m *= self.mined_loss_weight and (1. + self.mined_loss_weight) / self.mined_loss_weight
                if self.mined_loss_weight > 0:
                    keep_ratio = keep_mask.sum() / keep_mask.size(0)
                    self.writer.add_scalar('miner/keep_ratio', keep_ratio, self.step)
                    if self.scale_threshold:
                        loss_m /= keep_ratio
                loss_m = None if loss_m == 0. else loss_m
                self.log_scalars(loss=loss, loss_m=loss_m)

            # log miner stats
            if self.mined_loss_weight > 0 and self.rank == 0 and self.step % (self.total_steps // 100) == 0:  # currently self.log_steps
                # todo add cifar stats
                if not self.distributed:
                    # todo broadcast labels in distributed mode
                    self.log_miner_stats(selection_mask, track_across_steps=self.num_examples == 1)
                    self.log_miner_dist(mined_dist)

                    if 'metadata_m' in inputs and 'metadata_pool' in inputs:
                        labels = inputs['metadata_m']['labels']
                        labels_pool = inputs['metadata_pool']['labels']
                        if labels == 'data.y' and labels_pool == 'data.y':
                            # labels are in the graph
                            labels, labels_pool = view_m.y.cpu().numpy(), view_pool.y.cpu().numpy()
                        # todo fix this add log_mask
                        self.log_miner_confusion_matrix(labels, labels_pool,
                                                        selection_mask.cpu().numpy(), comment='_pre_thresh')
                        self.log_miner_confusion_matrix(labels[keep_mask.cpu().numpy()], labels_pool,
                                                        selection_mask[keep_mask].cpu().numpy())

            # log images
            if self.log_img and self.step % self.log_img_steps:
                if self.rank == 0:
                    self.log_views(view_1, view_2, 'Augmented Views')

                if self.rank == 0 and self.mined_loss_weight > 0:
                        # todo update with threshold
                        # look for images that are on gpu 0
                        log_mask = selection_mask[:self.train_batch_size] < self.train_batch_size
                        log_mask_m = selection_mask[:self.train_batch_size][log_mask]
                        self.log_views(view_m[log_mask], view_pool[log_mask_m], 'Mined Views')

                """
                # not enough, need to broadcast
                if self.distributed:
                    # get image pools from all gpus
                    # can get expensive
                    gather_list = [torch.zeros_like(view_pool) for _ in range(self.world_size)]
                    dist.all_gather(gather_list, view_pool, self.group)
                    view_pool = torch.cat(gather_list, dim=0)

                if self.rank == 0:
                    self.log_views(view_m, view_pool[selection_mask], 'Mined Views')
                """

            # update step
            self.step += 1
        self.epoch += 1
