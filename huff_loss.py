import torch
import torch.nn.functional as F
from torch import Tensor
from functools import lru_cache


def _soft_threshold(values: Tensor, threshold: float = 0.5) -> Tensor:
    # tf: sign(x) * relu(abs(x) - threshold)
    return torch.sign(values) * F.relu(values.abs() - threshold)


@lru_cache(maxsize=None)
def _hausdorff_kernel(radius: int, device_str: str, dtype_str: str) -> Tensor:
    """
    TF code uses:
      xy = linspace(-r, r, (2*r)-1)
      kernel = (sqrt(x^2+y^2) <= r) / n
    NOTE: That yields size (2r-1) x (2r-1) (not 2r+1).
    We'll match it exactly.
    """
    r = int(radius)
    k = 2 * r - 1
    xy = torch.linspace(-r, r, steps=k)
    x, y = torch.meshgrid(xy, xy, indexing="ij")
    rr = torch.sqrt(x * x + y * y)
    mask = (rr <= float(r)).to(torch.float32)
    kernel = mask / (mask.sum() + 1e-8)
    # conv2d expects [out_ch, in_ch, kH, kW]
    kernel = kernel.view(1, 1, k, k)
    return kernel


def _get_kernel(radius: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    k = _hausdorff_kernel(radius, str(device), str(dtype))
    return k.to(device=device, dtype=dtype)


def _hausdorff_loss_one_radius(
    p: Tensor, p_c: Tensor, q: Tensor, q_c: Tensor,
    f_qp: Tensor, f_pq: Tensor,
    radius: int,
) -> Tensor:
    """
    Mirrors TF hausdorff_loss():
      Br = kernel(radius)
      Or1 = soft(conv(p_c,Br)) * f_qp
      Or2 = soft(conv(p,Br))   * f_pq
      Or3 = soft(conv(q_c,Br)) * f_pq
      Or4 = soft(conv(q,Br))   * f_qp
      Lr = mean(Or1 + Or2 + Or3 + Or4)
    All tensors are [N,1,H,W].
    """
    Br = _get_kernel(radius, device=p.device, dtype=p.dtype)
    # TF padding='SAME' => PyTorch padding = floor(k/2)
    pad = (Br.shape[-1] // 2)

    Cr1 = F.conv2d(p_c, Br, stride=1, padding=pad)
    Or1 = _soft_threshold(Cr1, threshold=0.5) * f_qp

    Cr2 = F.conv2d(p, Br, stride=1, padding=pad)
    Or2 = _soft_threshold(Cr2, threshold=0.5) * f_pq

    Cr3 = F.conv2d(q_c, Br, stride=1, padding=pad)
    Or3 = _soft_threshold(Cr3, threshold=0.5) * f_pq

    Cr4 = F.conv2d(q, Br, stride=1, padding=pad)
    Or4 = _soft_threshold(Cr4, threshold=0.5) * f_qp

    return (Or1 + Or2 + Or3 + Or4).mean()


def hausdorff_dice_loss_slitnet_tf_equiv(
    target: Tensor,  # [N,H,W] 0/1
    output: Tensor,  # [N,H,W] probabilities in [0,1]
    power: float = 2.0,  # config.MASK_HAUSDORFF_POWER (paper uses r^2) :contentReference[oaicite:1]{index=1}
) -> Tensor:
    """
    Exact translation of your TF 'hausdorff_dice_loss(target, output)'.
    """
    # expand dims to [N,1,H,W]
    target = target.float().unsqueeze(1)
    output = output.float().unsqueeze(1)

    # dice (TF uses +1 smoothing in numerator and denominator)
    overlap = target * output
    intersection = overlap.sum(dim=(2, 3))
    union_target = (target * target).sum(dim=(2, 3))
    union_output = (output * output).sum(dim=(2, 3))
    dice = (1.0 - ((2.0 * intersection + 1.0) / (union_target + union_output + 1.0))).mean()

    # hausdorff pieces (TF uses thresholded versions of target/output)
    p = (target > 0.5).float()
    p_c = 1.0 - p
    q = (output > 0.5).float()
    q_c = 1.0 - q

    # TF:
    # f_qp = (target-output)^2 * output
    # f_pq = (output-target)^2 * target
    f_qp = (target - output).pow(2) * output
    f_pq = (output - target).pow(2) * target

    radii = (3, 6, 9, 12, 15, 18)
    hausdorff = 0.0
    for r in radii:
        hausdorff = hausdorff + (float(r) ** float(power)) * _hausdorff_loss_one_radius(
            p, p_c, q, q_c, f_qp, f_pq, r
        )

    # TF: hausdorff + (hausdorff/dice)*dice
    # which simplifies algebraically to 2*hausdorff, BUT we keep it verbatim to match TF numerics.
    loss = hausdorff + (hausdorff / (dice + 1e-8)) * dice
    return loss


def hausdorff_dice_loss_from_logits_slitnet_tf_equiv(
    mask_logits_gt: Tensor,  # [N,H,W] logits for GT class
    mask_targets: Tensor,    # [N,H,W] 0/1
    power: float = 2.0,
) -> Tensor:
    probs = torch.sigmoid(mask_logits_gt)
    return hausdorff_dice_loss_slitnet_tf_equiv(mask_targets, probs, power=power)