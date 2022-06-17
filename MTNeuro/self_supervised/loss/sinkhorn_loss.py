import torch
import torch.nn.functional as F


class SinkhornLoss(torch.nn.Module):
    r"""Sinkhorn loss.

    Args:
        gamma (float, optional): Regularization parameter.
        n_iters (float, optional): Number of iterations in the Sinkhorn algorithm.
    """
    def __init__(self, gamma=1e0, n_iters=10):
        super().__init__()

        self.gamma = gamma
        self.n_iters = n_iters

    def forward(self, outputs, targets, return_transport_plan=False):
        r"""Computes Sinkhorn distance between :obj:`outputs` and :obj:`targets`.
        If :obj:`return_transport_plan` is :obj:`True`, will return transport plan as well.
        """
        outputs = F.normalize(outputs, dim=-1, p=2)
        targets = F.normalize(targets, dim=-1, p=2)

        # compute cost matrix
        C = 2 - 2 * torch.sum(outputs.view(outputs.shape[0], 1, outputs.shape[1]) *
                              targets.view(1, targets.shape[0], targets.shape[1]), -1)

        # get regularized cost matrix
        K = torch.exp(-C / self.gamma)

        # run the Sinkhron algorithm
        a = torch.full([outputs.shape[0]], 1. / outputs.shape[0], requires_grad=True).to(outputs.device)
        b = torch.full([targets.shape[0]], 1. / targets.shape[0], requires_grad=True).to(outputs.device)
        lefts = [torch.ones_like(a)]
        rights = []

        for i in range(self.n_iters):
            rights += [b / torch.matmul(lefts[i - 1], K)]
            lefts += [a / torch.matmul(K, rights[i])]

        # get transport plan
        transport_plan = lefts[-1].view(-1, 1) * K * rights[-1].view(1, -1)

        if return_transport_plan:
            return torch.sum(C * transport_plan), transport_plan
        else:
            return torch.sum(C * transport_plan)
