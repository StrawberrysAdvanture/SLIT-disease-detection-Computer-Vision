from typing import List, Tuple

import torch
import torchvision
from torch import Tensor
from torch import Tensor
from torchvision.models.detection.rpn import RegionProposalNetwork
import torch.nn.functional as F
from losses.focal import binary_focal_loss_with_logits




import torch
import torch.nn.functional as F
from torch import Tensor

class SLITRPN(RegionProposalNetwork):
    def compute_loss(
            self, objectness: Tensor, pred_bbox_deltas: Tensor, labels: List[Tensor], regression_targets: List[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]
        
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        
        objectness = objectness.flatten()
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        
        num_samples = max(sampled_inds.numel(), 1)
        box_loss = F.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1 / 9,
            reduction="sum",
        ) / num_samples
        
        objectness_loss = binary_focal_loss_with_logits(
            objectness[sampled_inds],
            labels[sampled_inds].to(dtype=objectness.dtype),
            gamma=2.0,
        )
        
        return objectness_loss, box_loss