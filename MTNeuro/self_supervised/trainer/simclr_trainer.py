
from MTNeuro.self_supervised.models import SimCLR, MLP3
from MTNeuro.self_supervised.trainer import BYOLTrainer
from MTNeuro.self_supervised.loss import ContrastiveLoss


class SimCLRTrainer(BYOLTrainer):
    def __init__(self, temperature=0.1, **kwargs):
        super().__init__(**kwargs)

        self.loss = ContrastiveLoss(temperature).to(self.device)

    def _build_model(self, encoder):
        projector = MLP3(self.representation_size, self.projector_output_size, self.projector_hidden_size)
        net = SimCLR(encoder, projector)
        return net.to(self.device)

    def forward_loss(self, queries, keys):
        loss = self.loss(queries, keys)
        return loss

    def train_epoch(self):
        self.model.train()
        for inputs in self.train_dataloader:
            # update parameters
            self.update_learning_rate(self.step)
            self.update_momentum(self.step)

            inputs = self.prepare_views(inputs)
            queries = inputs['queries'].to(self.device)
            keys = inputs['keys'].to(self.device)

            if self.transform_1 is not None:
                # apply transforms
                queries = self.transform_1(queries)
                keys = self.transform_2(keys)

            # forward
            queries, keys = self.model(queries, keys)
            loss = self.forward_loss(queries, keys)

            # backprop online network
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # log
            if self.step % self.log_steps == 0 and self.rank == 0:
                self.log_scalars(loss=loss.item())

            # update parameters
            self.step += 1
