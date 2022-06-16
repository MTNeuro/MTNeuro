import torch
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, queries, keys):
        b, device = queries.shape[0], queries.device
        logits = queries @ keys.t()
        logits = logits - logits.max(dim=-1, keepdim=True).values
        logits /= self.temperature
        return F.cross_entropy(logits, torch.arange(b, device=device))
