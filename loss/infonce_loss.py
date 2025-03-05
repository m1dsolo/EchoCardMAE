import torch
from torch.nn import functional as F

def infoNCE_loss(q, k, t: float = 0.2):
    q = F.normalize(q, dim=1)
    k = F.normalize(k, dim=1)

    logits = q @ k.T
    labels = torch.arange(q.shape[0], device=q.device, dtype=torch.long)
    loss = F.cross_entropy(logits / t, labels, reduction='mean')

    return loss
