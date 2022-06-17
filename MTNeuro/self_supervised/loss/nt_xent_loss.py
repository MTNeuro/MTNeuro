import torch
import torch.nn.functional as F


class NtXentLoss(torch.nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, queries, keys):
        b, device = queries.shape[0], queries.device

        n = b * 2
        projs = torch.cat((queries, keys))
        logits = projs @ projs.t()

        mask = torch.eye(n, device=device).bool()
        logits = logits[~mask].reshape(n, n - 1)
        logits /= self.temperature

        labels = torch.cat(((torch.arange(b, device=device) + b - 1), torch.arange(b, device=device)), dim=0)
        loss = F.cross_entropy(logits, labels, reduction='sum')
        loss /= 2 * (b - 1)
        return loss
