import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator
from models.rpn import SLITRPN
from models.roi_heads import SLITRoIHeads
from torchvision.models.detection.rpn import RPNHead

def build_base_maskrcnn(num_classes: int, pretrained_backbone: bool = True):
    weights = None
    weights_backbone = "DEFAULT" if pretrained_backbone else None

    model = maskrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=weights_backbone,
        num_classes=num_classes,
    )

    anchor_gen = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)),
        aspect_ratios=((0.25, 0.5, 1.0, 2.0, 4.0),) * 6,  # NOTE: 6 levels since you provided 6 sizes
    )

    # 1) set anchor generator
    model.rpn.anchor_generator = anchor_gen

    # 2) rebuild RPN head with correct number of anchors per location
    out_channels = model.backbone.out_channels
    num_anchors = anchor_gen.num_anchors_per_location()[0]  # should be 5
    model.rpn.head = RPNHead(out_channels, num_anchors)

    return model

def replace_rpn_with_slit(model):
    # Reuse anchor_generator and head; swap the RPN module so compute_loss is yours
    old = model.rpn
    model.rpn = SLITRPN(
        anchor_generator=old.anchor_generator,
        head=old.head,
        fg_iou_thresh=old.proposal_matcher.high_threshold,
        bg_iou_thresh=old.proposal_matcher.low_threshold,
        batch_size_per_image=old.fg_bg_sampler.batch_size_per_image,
        positive_fraction=old.fg_bg_sampler.positive_fraction,
        pre_nms_top_n=old._pre_nms_top_n,
        post_nms_top_n=old._post_nms_top_n,
        nms_thresh=old.nms_thresh,
        score_thresh=old.score_thresh,
    )
    return model

def replace_roi_heads_with_slit(model):
    old = model.roi_heads
    
    model.roi_heads = SLITRoIHeads(
        box_roi_pool=old.box_roi_pool,
        box_head=old.box_head,
        box_predictor=old.box_predictor,
        fg_iou_thresh=old.proposal_matcher.high_threshold,
        bg_iou_thresh=old.proposal_matcher.low_threshold,
        batch_size_per_image=old.fg_bg_sampler.batch_size_per_image,
        positive_fraction=old.fg_bg_sampler.positive_fraction,
        bbox_reg_weights=old.box_coder.weights,  # important
        score_thresh=old.score_thresh,
        nms_thresh=old.nms_thresh,
        detections_per_img=old.detections_per_img,
        mask_roi_pool=old.mask_roi_pool,
        mask_head=old.mask_head,
        mask_predictor=old.mask_predictor,
    )
    return model

def build_slit_maskrcnn(num_classes: int, pretrained_backbone: bool = True):
    model = build_base_maskrcnn(num_classes, pretrained_backbone=pretrained_backbone)
    model = replace_rpn_with_slit(model)
    model = replace_roi_heads_with_slit(model)
    return model

if __name__ == "__main__":
    model = build_slit_maskrcnn(num_classes=1+1, pretrained_backbone=True)
    model.train()
    
    images = [torch.randn(3, 512, 512)]
    targets = [{
        "boxes": torch.tensor([[10., 10., 100., 120.]]),
        "labels": torch.tensor([1]),
        "masks": torch.zeros(1, 512, 512).to(torch.uint8),
    }]
    
    losses = model(images, targets)
    print(losses.keys(), sum(losses.values()))
