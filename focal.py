import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

def binary_focal_loss_with_logits(
    logits: Tensor,
    targets: Tensor,
    gamma: float = 2.0,
    alpha: Optional[float] = None,
    reduction: str = "mean",
) -> Tensor:
    """
    logits: [N] or any shape
    targets: same shape, 0/1
    """
    targets = targets.to(dtype=logits.dtype)
    prob = torch.sigmoid(logits)
    pt = torch.where(targets == 1, prob, 1 - prob)
    loss = -(1 - pt) ** gamma * torch.log(pt.clamp_min(1e-8))

    if alpha is not None:
        at = torch.where(targets == 1, alpha, 1 - alpha)
        loss = at * loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    raise ValueError(f"Invalid reduction: {reduction}")

def focal_loss_multiclass(
    logits: Tensor,
    targets: Tensor,
    gamma: float = 2.0,
    alpha: Optional[Tensor] = None,
    reduction: str = "mean",
) -> Tensor:
    """
    logits: [N, C]
    targets: [N] int64 in [0, C-1]
    alpha: optional per-class weights, shape [C]
           (set to None to match SLIT-Net style)
    """
    targets = targets.long()
    logp = F.log_softmax(logits, dim=1)
    p = logp.exp()
    idx = torch.arange(logits.size(0), device=logits.device)

    logpt = logp[idx, targets]
    pt = p[idx, targets]
    loss = -(1 - pt) ** gamma * logpt

    if alpha is not None:
        loss = alpha[targets] * loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    raise ValueError(f"Invalid reduction: {reduction}")

