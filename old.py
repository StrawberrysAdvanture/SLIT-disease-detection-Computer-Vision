# pip install torch torchvision opencv-python scikit-image
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import roi_align
import torchvision
import math
import numpy as np
from torch.utils.data import Dataset
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import time
from collections import defaultdict
import matplotlib
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.transforms.functional import to_pil_image
from torchvision.ops import clip_boxes_to_image
import os
from torchvision.models.detection.roi_heads import project_masks_on_boxes

from mask_rcnn import build_slit_maskrcnn
LABEL_MAP = {
    'pupil'       : 1,
    'cornea'      : 2,
    'uppereyelid' : 3,
    'lowereyelid' : 4,
    'lightreflex' : 5,
    'pinguecula'  : 6,
    'conjnevus'   : 7,
    'pseudophakia': 8,
    'cataract'    : 9,
    'irisnevus'   : 10,
}
'''
def smooth_l1_loss(input, target, beta: float = 1/9, reduction="mean"):
    diff = torch.abs(input - target)
    loss = torch.where(diff < beta, 0.5 * diff**2 / beta, diff - 0.5 * beta)
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss

def sigmoid_focal_loss(logits, targets, gamma=2.0, alpha=None, reduction="mean"):
    # logits: (N,) raw logits; targets: (N,) in {0,1}
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = (1 - p_t) ** gamma * ce
    if alpha is not None:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean() if reduction == "mean" else loss.sum()

def softmax_focal_loss(logits, targets, gamma=2.0, reduction="mean"):
    # logits: (N, C); targets: (N,) int64
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)  # prob of the true class
    loss = (1 - pt) ** gamma * ce
    return loss.mean() if reduction == "mean" else loss.sum()

def rpn_box_loss(pred_bbox_deltas, gt_bbox_deltas, labels, beta=1/9):
    # pred_bbox_deltas: (N,4); gt_bbox_deltas: (N,4); labels: (N,) with 1=pos, 0=neg, -1=ignore
    labels = labels.view(-1)
    pos = torch.where(labels == 1)[0]
    if pos.numel() == 0:
        return pred_bbox_deltas.sum() * 0.0
    return smooth_l1_loss(pred_bbox_deltas[pos], gt_bbox_deltas[pos], beta=beta, reduction="sum") / labels.numel()



class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma, self.alpha, self.reduction = gamma, alpha, reduction

    def forward(self, logits, targets):
        """
        logits: (N, C) raw logits
        targets: (N,) long with class indices
        """
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)                 # pt = softmax prob of the true class
        focal = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            at = self.alpha[targets] if isinstance(self.alpha, torch.Tensor) else self.alpha
            focal = at * focal
        return focal.mean() if self.reduction == 'mean' else focal.sum()


def soft_dice_loss(pred, target, eps=1e-6):
    """
    pred, target: (N, 1, H, W) probabilities in [0,1] and binary GT
    """
    num = 2 * (pred * target).sum(dim=(2,3)) + eps
    den = (pred**2).sum(dim=(2,3)) + (target**2).sum(dim=(2,3)) + eps
    dice = 1 - (num / den)
    return dice.mean()

def make_circular_kernel(radius):
    d = 2*radius+1
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x*x + y*y <= radius*radius
    k = np.zeros((d, d), dtype=np.float32)
    k[mask] = 1.0
    k /= k.sum()
    k = torch.from_numpy(k).unsqueeze(0).unsqueeze(0)  # (1,1,d,d)
    return k

class HausdorffDiceLoss(nn.Module):
    def __init__(self, radii=(3,6,9,12,15,18)):
        super().__init__()
        self.radii = radii
        self.kernels = nn.ParameterList(
            [nn.Parameter(make_circular_kernel(r), requires_grad=False) for r in radii]
        )

    @staticmethod
    def _soft_thresh(x, t=0.5):
        # sign(x) * max(|x|-t, 0)
        return torch.sign(x) * torch.clamp(x.abs() - t, min=0.)

    def forward(self, pred, target):
        """
        pred: (N,1,H,W) after sigmoid; target: (N,1,H,W) in {0,1}
        """
        # Dice term
        dice = soft_dice_loss(pred, target)

        # Hausdorff proxy term
        # q = pred, q_bin = (pred>0.5), m = target
        q = pred
        m = target
        q_bin = (q > 0.5).float()
        m_comp = 1 - m
        q_comp = 1 - q_bin

        haus = 0.0
        N = pred.size(0)
        for k in self.kernels:
            k = k.to(pred.device)
            # convolutions
            Bm  = F.conv2d(m,  k, padding=k.shape[-1]//2)
            Bqb = F.conv2d(q_bin, k, padding=k.shape[-1]//2)
            # soft-thresholded maps
            s1 = self._soft_thresh(Bm) * ((q - m)**2 * (q > m).float())   # fsoft(Br*m) ◦ f_{q\m}
            s2 = self._soft_thresh(Bqb) * ((m - q)**2 * (m > q).float())  # fsoft(Br*q̄) ◦ f_{m\q}
            haus = haus + s1.mean() + s2.mean()

        # balance λ so both terms contribute similarly (paper equalizes magnitudes) :contentReference[oaicite:5]{index=5}
        lam = haus.detach() / (dice.detach() + 1e-6)
        return haus + lam * dice
    
    
def build_slitnet(num_classes):
    # num_classes includes background; map classes as you prefer
    anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )
    model = maskrcnn_resnet50_fpn(weights="DEFAULT",
                                  rpn_anchor_generator=anchor_generator,
                                  box_detections_per_img=1000)  # keep many, we’ll filter later

    # Replace the box predictor head to match num_classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor.cls_score = nn.Linear(in_features, num_classes)
    model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features, num_classes* 4)

    # Replace mask predictor to output (num_classes - 1) channels
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden = 256
    model.roi_heads.mask_predictor = \
        torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
            in_features_mask, hidden, num_classes-1
        )

    # Plug in our custom losses
    model.slitnet_focal = FocalLoss(gamma=2.0)
    model.slitnet_mask_loss = HausdorffDiceLoss()

    # --- Monkey-patch ROI classification loss ---
    
    def fastrcnn_loss_focal(class_logits, box_regression, labels, regression_targets):
        targets = torch.cat(labels, dim=0)  # (N,)
        reg_targets = torch.cat(regression_targets, dim=0)  # (N,4)
        
        cls_loss = softmax_focal_loss(class_logits, targets, gamma=2.0, reduction="mean")
        
        pos = torch.where(targets > 0)[0]
        if pos.numel() == 0:
            return cls_loss, box_regression.sum() * 0.0
        
        N, C = class_logits.shape
        pos_labels = targets[pos]
        valid = (pos_labels >= 1) & (pos_labels <= C - 1)
        pos = pos[valid]
        if pos.numel() == 0:
            return cls_loss, box_regression.sum() * 0.0
        
        if box_regression.shape[1] == 4:  # class-agnostic
            pos_regression = box_regression[pos]
        else:  # class-specific
            assert box_regression.shape[1] == C * 4, f"Expected {C * 4}, got {box_regression.shape[1]}"
            box_regression = box_regression.view(N, C, 4)
            pos_labels = targets[pos]
            pos_regression = box_regression[pos, pos_labels]  # safe after valid-filter
        
        pos_targets = reg_targets[pos]
        box_loss = smooth_l1_loss(pos_regression, pos_targets, beta=1 / 9, reduction="sum")
        box_loss = box_loss / targets.numel()
        return cls_loss, box_loss
    
    # activate the patch
    
    # --- Monkey-patch RPN classification loss to focal ---
    

    # activate the patch

    
    # --- Monkey-patch mask loss to Hausdorff-Dice ---
    def mask_loss_hausdorff_dice(mask_logits, proposals, gt_masks, gt_labels, pos_matched_idxs):
        total_loss, count = 0.0, 0
        
        for logits_i, boxes_i, gt_masks_i, gt_labels_i, matched in zip(
                mask_logits, proposals, gt_masks, gt_labels, pos_matched_idxs
        ):
            if logits_i.numel() == 0:
                continue
            
            device = logits_i.device
            matched = matched.to(device=device, dtype=torch.long)
            
            # P from each source
            P_logit = logits_i.shape[0]
            P_box = boxes_i.shape[0]
            P_match = matched.numel()
            
            # sync to common P
            P = min(P_logit, P_box, P_match)
            if P == 0:
                continue
            logits_i = logits_i[:P]
            boxes_i = boxes_i[:P]
            matched = matched[:P]
            
            # dims
            if logits_i.dim() == 4:  # (P, K, M, M)
                _, K, M, _ = logits_i.shape
            elif logits_i.dim() == 3:  # (P, M, M)
                K = None
                _, M, _ = logits_i.shape
            else:
                continue
            
            # validity checks to avoid OOB on GPU
            G = gt_masks_i.shape[0]
            b = boxes_i
            valid_box = torch.isfinite(b).all(dim=1) & (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
            valid_match = (matched >= 0) & (matched < G)
            valid = valid_box & valid_match
            
            if K is not None:
                gl_dev = gt_labels_i.to(device=device, dtype=torch.long)
                labels_pos_full = gl_dev[matched]  # (P,)
                valid = valid & (labels_pos_full >= 1) & (labels_pos_full <= K)
            
            keep = torch.nonzero(valid, as_tuple=False).squeeze(1)
            if keep.numel() == 0:
                continue
            
            logits_i = logits_i[keep]
            boxes_i = boxes_i[keep]
            matched = matched[keep]
            P = logits_i.shape[0]
            
            # project GT masks to RoIs @ MxM (tv 0.17.2: 4-arg API)
            tgt = project_masks_on_boxes(
                gt_masks_i.to(device=device),  # (G,H,W)
                boxes_i.to(device),  # (P,4)
                matched,  # (P,)
                M  # int
            )  # -> (P,M,M)
            tgt = tgt.unsqueeze(1).float()  # (P,1,M,M)
            
            # predictions (class-agnostic vs class-specific)
            if K is None:
                pred = torch.sigmoid(logits_i).unsqueeze(1)  # (P,1,M,M)
            else:
                labels_pos = gt_labels_i.to(device=device, dtype=torch.long)[matched]
                labels_pos = labels_pos.clamp_(1, K)  # safety
                chosen = logits_i[torch.arange(P, device=device), labels_pos - 1]
                pred = torch.sigmoid(chosen).unsqueeze(1)  # (P,1,M,M)
            
            total_loss += model.slitnet_mask_loss(pred, tgt)
            count += 1
        
        return total_loss / max(count, 1)
    
    def rpn_loss_focal(objectness, pred_bbox_deltas, labels, regression_targets):
        obj_logits = torch.cat(objectness, dim=0)  # (N,)
        lbls = torch.cat(labels, dim=0)  # {1,0,-1}
        deltas = torch.cat(pred_bbox_deltas, dim=0)  # (N,4)
        deltas_tgt = torch.cat(regression_targets, dim=0)  # (N,4)
        
        keep = torch.where(lbls >= 0)[0]  # ignore -1
        cls_loss = sigmoid_focal_loss(obj_logits[keep], lbls[keep], gamma=2.0, reduction="mean")
        box_loss = rpn_box_loss(deltas, deltas_tgt, lbls, beta=1 / 9)
        return cls_loss, box_loss
    
    def safe_maskrcnn_inference(mask_logits, labels):
        """
        mask_logits: List[Tensor], each (N,K,M,M) or (N,M,M) or (N,1,M,M)
        labels:      List[Tensor], each (N,) predicted class ids
        returns:     List[Tensor], each (N,1,M,M) mask probabilities
        """
        out = []
        for logit_i, labels_i in zip(mask_logits, labels):
            prob = logit_i.sigmoid()  # to [0,1]
            device = prob.device
            
            # Normalize labels_i to a LongTensor on the right device
            if torch.is_tensor(labels_i):
                lbl = labels_i.to(device=device, dtype=torch.long)
            else:
                lbl = torch.as_tensor(labels_i, device=device, dtype=torch.long)
            
            # Class-agnostic shapes
            if prob.dim() == 3:  # (N,M,M)
                N = prob.shape[0]
                out.append(prob.unsqueeze(1))  # -> (N,1,M,M)
                continue
            if prob.dim() == 4 and prob.shape[1] == 1:  # (N,1,M,M)
                out.append(prob)  # already (N,1,M,M)
                continue
            
            # Class-specific: (N,K,M,M)
            if prob.dim() != 4:
                # Unexpected—return a safe fallback
                out.append(prob[:, :1] if prob.dim() > 1 else prob.unsqueeze(1))
                continue
            
            N, K, M, _ = prob.shape
            
            # Sync lengths if needed
            if lbl.numel() != N:
                N2 = min(N, lbl.numel())
                prob = prob[:N2]
                lbl = lbl[:N2]
                N = N2
                if N == 0:
                    out.append(prob[:, :1])
                    continue
            
            # Convert 1-based labels (1..K) to 0-based (0..K-1) when appropriate
            if lbl.numel() > 0 and (lbl.min() >= 1) and (lbl.max() <= K):
                lbl = lbl - 1
            
            # Clamp to avoid OOB
            if K > 0:
                lbl = lbl.clamp(0, K - 1)
            
            rows = torch.arange(N, device=device)
            sel = prob[rows, lbl]  # (N,M,M)
            out.append(sel.unsqueeze(1))  # (N,1,M,M)
        
        return out
    
    import torchvision.models.detection.roi_heads as tv_rh
    import torchvision.models.detection.rpn as tv_rpn
    tv_rh.fastrcnn_loss = fastrcnn_loss_focal
    tv_rh.maskrcnn_loss = mask_loss_hausdorff_dice
    tv_rpn.rpn_loss = rpn_loss_focal
    tv_rh.maskrcnn_inference = safe_maskrcnn_inference
    
    return model
'''
ABNORMAL_SET = {"pinguecula", "conjnevus", "pseudophakia", "cataract", "irisnevus"}
def filter_images_with_abnormal(items, abnormal_set=ABNORMAL_SET):
    """Keep only images that contain at least one polygon with an abnormal label."""
    kept = []
    for rec in items:
        has_abnormal = any(poly['label'] in abnormal_set for poly in rec['polys'])
        if has_abnormal:
            kept.append(rec)
    return kept

def parse_cvat_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    images = []
    for img in root.iter('image'):
        name = img.attrib['name']
        W, H = int(img.attrib['width']), int(img.attrib['height'])
        polys = []
        for poly in img.iter('polygon'):
            label = poly.attrib['label']        # e.g., "cornea"
            pts = []
            for p in poly.attrib['points'].split(';'):
                if not p.strip(): continue
                x,y = p.split(',')
                pts.append((float(x), float(y)))
            polys.append({'label': label, 'points': pts})
        images.append({'name': name, 'width': W, 'height': H, 'polys': polys})
    return images

def polygon_to_mask(points, H, W):
    img = Image.new('L', (W, H), 0)
    ImageDraw.Draw(img).polygon(points, outline=1, fill=1)
    return np.array(img, dtype=np.uint8)

def mask_to_box(mask):
    ys, xs = np.where(mask > 0)
    if ys.size == 0: return None
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    return [float(x1), float(y1), float(x2), float(y2)]
def resize_and_pad(img, masks, target_long=512):
    H, W = img.shape[:2]
    if H >= W:
        new_h = target_long; new_w = int(W * target_long / H)
    else:
        new_w = target_long; new_h = int(H * target_long / W)
    img_r = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((target_long, target_long, 3), dtype=img.dtype)
    canvas[:new_h, :new_w] = img_r

    out_masks = []
    for m in masks:
        m_r = cv2.resize(m, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        m_c = np.zeros((target_long, target_long), dtype=m.dtype)
        m_c[:new_h, :new_w] = m_r
        out_masks.append(m_c)
    return canvas, out_masks
'''
class CVATPolygonDataset(Dataset):
    def __init__(self, xml_path, image_root, augment=False, only_abnormal_polys=False):
        self.items = parse_cvat_xml(xml_path)
        # Keep only images that have ≥1 abnormal polygon:
        self.items = filter_images_with_abnormal(self.items, ABNORMAL_SET)

        self.root = Path(image_root)
        self.augment = augment
        self.only_abnormal_polys = only_abnormal_polys
        clean = []
        for rec in self.items:
            # try both full path and basename
            cand = self.root / rec['name']
            cand2 = self.root / Path(rec['name']).name
            if cand.exists() or cand2.exists():
                clean.append(rec)
            else:
                print(f"[WARN] Skipping missing file: {rec['name']}")
        self.items = clean
        
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, i):
        rec = self.recs[i]  # whatever you called your parsed items
        
        # --- load image (BGR) ---
        path = self.root / rec['name']
        img_bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            raise FileNotFoundError(f"Missing image: {path}")
        
        # --- normalize channels & BGR->RGB without negative strides ---
        if img_bgr.ndim == 2:  # grayscale -> 3ch
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        elif img_bgr.shape[2] == 4:  # drop alpha if present
            img_bgr = img_bgr[:, :, :3]
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # no negative strides
        img_rgb = np.ascontiguousarray(img_rgb)  # make sure contiguous
        
        img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        H, W = img_t.shape[-2:]
        
        masks, boxes, labels = [], [], []
        for ann in rec['anns']:
            cls_id = ann['cls_id']
            poly = np.asarray(ann['poly'], dtype=np.float32)  # [[x,y],...]
            
            # (optional) clamp polygons to image bounds
            poly[:, 0] = np.clip(poly[:, 0], 0, W - 1)
            poly[:, 1] = np.clip(poly[:, 1], 0, H - 1)
            
            # rasterize at original size
            m = polygon_to_mask(poly, H, W)  # (H,W) uint8/bool
            if m is None or np.count_nonzero(m) == 0:
                continue
            
            # box directly from polygon min/max (simpler & safer than scanning mask)
            x1, y1 = float(poly[:, 0].min()), float(poly[:, 1].min())
            x2, y2 = float(poly[:, 0].max()), float(poly[:, 1].max())
            
            boxes.append(torch.tensor([x1, y1, x2, y2], dtype=torch.float32))
            labels.append(torch.tensor(cls_id, dtype=torch.int64))
            masks.append(torch.from_numpy(np.ascontiguousarray(m)))  # ensure contiguous
        
        if boxes:
            boxes_t = torch.stack(boxes)
            labels_t = torch.stack(labels)
            masks_t = torch.stack(masks).to(torch.uint8)  # [N,H,W]
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            masks_t = torch.zeros((0, H, W), dtype=torch.uint8)
        
        target = {"boxes": boxes_t, "labels": labels_t, "masks": masks_t}
        return img_t, target
    '''
class CVATPolygonDataset(Dataset):
    def __init__(self, xml_path, image_root, augment=False, only_abnormal_polys=False):
        self.items = parse_cvat_xml(xml_path)                        # -> uses 'polys'
        self.items = filter_images_with_abnormal(self.items, ABNORMAL_SET)
        self.root = Path(image_root)
        self.augment = augment
        self.only_abnormal_polys = only_abnormal_polys

        # drop missing files
        clean = []
        for rec in self.items:
            p1 = self.root / rec['name']
            p2 = self.root / Path(rec['name']).name
            if p1.exists() or p2.exists():
                clean.append(rec)
            else:
                print(f"[WARN] Skipping missing file: {rec['name']}")
        self.items = clean

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]

        # ---- read image safely (no negative strides) ----
        p1 = self.root / rec['name']
        p2 = self.root / Path(rec['name']).name
        path = p1 if p1.exists() else p2
        img_bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            raise FileNotFoundError(f"Missing image: {path}")

        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        elif img_bgr.shape[2] == 4:
            img_bgr = img_bgr[:, :, :3]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = np.ascontiguousarray(img_rgb)

        img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        H, W = img_t.shape[-2:]

        # ---- build targets from rec["polys"] ----
        boxes, labels, masks = [], [], []
        polys = rec["polys"]
        if self.only_abnormal_polys:
            polys = [p for p in polys if p["label"] in ABNORMAL_SET]

        for poly in polys:
            name = poly["label"]
            if name not in LABEL_MAP:
                # skip unknown labels
                continue
            cls_id = LABEL_MAP[name]

            pts = np.asarray(poly["points"], dtype=np.float32)  # [[x,y],...]
            # clamp into image bounds
            pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)

            m = polygon_to_mask([tuple(p) for p in pts.tolist()], H, W)  # (H,W) uint8
            if m is None or not np.any(m):
                continue

            x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
            x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
            boxes.append(torch.tensor([x1, y1, x2, y2], dtype=torch.float32))
            labels.append(torch.tensor(cls_id, dtype=torch.int64))
            masks.append(torch.from_numpy(np.ascontiguousarray(m.astype(np.uint8))))

        if boxes:
            boxes_t  = torch.stack(boxes)
            labels_t = torch.stack(labels)
            masks_t  = torch.stack(masks)  # [N,H,W], uint8
        else:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            masks_t  = torch.zeros((0, H, W), dtype=torch.uint8)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "masks": masks_t,
            "image_id": torch.tensor([idx]),
        }
        return img_t, target


def show_sample(img, target, label_map=LABEL_MAP, alpha=0.4):
    """
    img: torch.Tensor (3,H,W) normalized [0,1]
    target: dict with 'boxes', 'labels', 'masks'
    label_map: dict (str->int)
    alpha: transparency for masks
    """
    # Convert tensor to numpy for plotting
    img_np = img.permute(1,2,0).cpu().numpy().copy()

    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(img_np)

    masks = target["masks"].cpu().numpy()
    boxes = target["boxes"].cpu().numpy()
    labels = target["labels"].cpu().numpy()

    # Reverse the label map (id → name)
    id_to_name = {v:k for k,v in label_map.items()}

    # Pick a colormap (tab20 has 20 distinct colors)
    cmap = cm.get_cmap("tab20", len(label_map)+1)

    for m, box, lab in zip(masks, boxes, labels):
        color = cmap(int(lab))  # RGBA tuple from colormap
        # overlay the mask
        ax.imshow(np.ma.masked_where(m==0, m), cmap=cmap, alpha=alpha,
                  vmin=0, vmax=len(label_map))
        # draw bounding box
        x1,y1,x2,y2 = box
        rect = patches.Rectangle((x1,y1), x2-x1, y2-y1,
                                 linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        # label text
        ax.text(x1, y1-2, id_to_name.get(int(lab), str(lab)),
                color="white", fontsize=8,
                bbox=dict(facecolor=color, alpha=0.6, edgecolor="none", pad=1))

    ax.axis("off")
    plt.show()
    
#def collate_fn(batch):
 #   imgs, tgts = list(zip(*batch))
  #  return list(imgs), list(tgts)
def collate_fn(batch): return tuple(zip(*batch))

@torch.inference_mode()
def predict_batch(model, loader, device, score_thresh=0.5, max_batches=1):
    model.eval()
    images_b, outs_b = None, None

    for i, (images, _) in enumerate(loader):
        if i >= max_batches:
            break
        images = [im.to(device) for im in images]
        outs = model(images)  # list of dicts

        # per-image filtering
        for j, out in enumerate(outs):
            # make sure we have masks before indexing
            N = out["scores"].shape[0]
            keep = (out["scores"] >= score_thresh)
            if keep.numel() != N:
                # guard: ensure 1D boolean of length N
                keep = keep.reshape(-1)[:N]
            # apply to all aligned fields
            for k in ("boxes", "labels", "scores", "masks"):
                if k in out and torch.is_tensor(out[k]) and out[k].shape[0] == N:
                    out[k] = out[k][keep]

            # squeeze masks to (n,H,W) and move to cpu
            if "masks" in out and out["masks"].numel() > 0:
                # torchvision returns (n,1,H,W)
                if out["masks"].dim() == 4 and out["masks"].shape[1] == 1:
                    out["masks"] = out["masks"].squeeze(1)
                out["masks"] = out["masks"].cpu()

            # move others to cpu for viz
            out["boxes"]  = out["boxes"].cpu()
            out["labels"] = out["labels"].cpu()
            out["scores"] = out["scores"].cpu()

        images_b, outs_b = [im.cpu() for im in images], outs

    return images_b, outs_b

@torch.inference_mode()
def preview_predictions(model, loader, device, N=8, score_thresh=0.5, alpha=0.4):
    model.eval()
    shown = 0
    for images, _ in loader:
        images = [im.to(device) for im in images]
        outs = model(images)  # list[dict], one per image

        for img_t, out in zip(images, outs):
            # per-image score filter
            keep = out["scores"] >= score_thresh if "scores" in out else None

            def _maybe_keep(t):
                if t is None or not torch.is_tensor(t):
                    return t
                if keep is None or t.shape[0] != keep.shape[0]:
                    return t
                return t[keep]

            viz_target = {
                "boxes":  _maybe_keep(out.get("boxes")),
                "labels": _maybe_keep(out.get("labels")),
                "masks":  _maybe_keep(out.get("masks")),
            }

            # torchvision returns masks as (n,1,H,W) — make (n,H,W) for your viz
            if viz_target["masks"] is not None and viz_target["masks"].dim() == 4 and viz_target["masks"].shape[1] == 1:
                viz_target["masks"] = viz_target["masks"].squeeze(1)

            # move to cpu for plotting
            img_cpu = img_t.cpu()
            for k in ("boxes", "labels", "masks"):
                if torch.is_tensor(viz_target.get(k)):
                    viz_target[k] = viz_target[k].cpu()

            show_sample(img_cpu, viz_target, LABEL_MAP, alpha=alpha)

            shown += 1
            if shown >= N:
                return

    
def draw_on_image(img_t, out, id_to_name, score_thresh=None, alpha=0.35,
                  box_thickness=3, font_scale=0.9, text_thickness=2, text_bg_alpha=0.65):
    img = (img_t.detach().cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
    overlay = img.copy()
    H, W = img.shape[:2]

    boxes  = out.get("boxes",  None)
    labels = out.get("labels", None)
    scores = out.get("scores", None)
    masks  = out.get("masks",  None)

    if boxes is None or labels is None:
        plt.imshow(img); plt.axis("off"); return

    boxes  = boxes.detach().cpu() if torch.is_tensor(boxes)  else boxes
    labels = labels.detach().cpu() if torch.is_tensor(labels) else labels
    scores = scores.detach().cpu() if (scores is not None and torch.is_tensor(scores)) else scores
    if masks is not None and torch.is_tensor(masks):
        masks = masks.detach().cpu()
        if masks.dim() == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]

    n_boxes  = boxes.shape[0]
    n_labels = labels.shape[0]
    n_scores = scores.shape[0] if scores is not None else n_boxes
    n_masks  = masks.shape[0]  if masks  is not None else n_boxes

    N = min(n_boxes, n_labels, n_scores, n_masks)
    if N == 0:
        plt.imshow(img); plt.axis("off"); return
    boxes, labels = boxes[:N], labels[:N]
    if scores is not None: scores = scores[:N]
    if masks  is not None: masks  = masks[:N]

    if score_thresh is not None and scores is not None:
        keep = (scores[:N] >= score_thresh).reshape(-1)
        boxes, labels, scores = boxes[keep], labels[keep], scores[keep]
        if masks is not None: masks = masks[keep]
        N = boxes.shape[0]

    for i in range(N):
        x1, y1, x2, y2 = boxes[i].round().int().tolist()
        x1 = int(np.clip(x1, 0, W-1)); x2 = int(np.clip(x2, 0, W-1))
        y1 = int(np.clip(y1, 0, H-1)); y2 = int(np.clip(y2, 0, H-1))
        if x2 <= x1 or y2 <= y1:
            continue

        # deterministic color per class (stable across runs)
        rng = np.random.default_rng(int(labels[i]) * 1009)
        color = tuple(int(c) for c in rng.integers(60, 255, size=3))

        # box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, box_thickness)

        # mask
        if masks is not None and masks.numel() > 0:
            m = masks[i] > 0.5
            if m.any():
                overlay[m.numpy()] = (0.6*np.array(color) + 0.4*overlay[m.numpy()]).astype(np.uint8)

        # label text with background
        name = id_to_name.get(int(labels[i]), str(int(labels[i])))
        txt  = f"{name}:{float(scores[i]):.2f}" if scores is not None else name
        _draw_label_with_bg(overlay, x1, y1, txt, color,
                            font_scale=font_scale, text_thickness=text_thickness, bg_alpha=text_bg_alpha)

    blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0.0)
    plt.imshow(blended); plt.axis("off")


def preview_gt_only(model, loader, device, id2name, save_dir="viz/gt_only", max_images=12):
    """
    Runs one or more batches through the model (preds ignored) and saves PNGs
    with ONLY the ground-truth labels drawn on the original images.

    If you still see the black band here, it's coming from the images/dataset.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    shown = 0

    for images, targets in loader:
        # forward pass just to mirror normal pipeline; we don't use outputs
        _ = model([im.to(device) for im in images])

        # draw GT on original images
        for img, tgt in zip(images, targets):
            H, W = img.shape[-2:]

            boxes  = tgt.get("boxes",  torch.empty(0, 4))
            labels = tgt.get("labels", torch.empty(0, dtype=torch.long))
            masks  = tgt.get("masks",  None)

            # clip boxes to image and move to cpu
            if torch.is_tensor(boxes) and boxes.numel() > 0:
                boxes = clip_boxes_to_image(boxes, (H, W)).cpu()
            else:
                boxes = torch.empty(0, 4)

            # labels -> text
            if torch.is_tensor(labels) and labels.numel() > 0:
                labels = labels.cpu()
                label_txt = [id2name.get(int(l), str(int(l))) for l in labels.tolist()]
            else:
                labels = torch.empty(0, dtype=torch.long)
                label_txt = []

            # normalize masks to [N,H,W] bool
            bool_masks = torch.empty(0, H, W, dtype=torch.bool)
            if masks is not None and torch.is_tensor(masks) and masks.numel() > 0:
                m = masks
                if m.dim() == 4 and m.shape[1] == 1:  # [N,1,H,W] -> [N,H,W]
                    m = m.squeeze(1)
                if m.dim() == 3 and m.shape[-2:] != (H, W):
                    m = F.interpolate(m.float().unsqueeze(1), (H, W), mode="nearest").squeeze(1)
                bool_masks = (m > 0.5).cpu()

            # draw on original image
            img_u8 = (img.clamp(0, 1) * 255).to(torch.uint8).cpu()
            vis = draw_bounding_boxes(
                image=img_u8,
                boxes=boxes,
                labels=label_txt if len(label_txt) == boxes.shape[0] else None,
                width=2,
            )
            if bool_masks.numel() > 0:
                vis = draw_segmentation_masks(vis, bool_masks, alpha=0.45)

            out_path = os.path.join(save_dir, f"gt_{shown:03d}.png")
            to_pil_image(vis).save(out_path)
            print(f"[gt-only] saved {out_path}")
            shown += 1

            if shown >= max_images:
                return

            
            
def _draw_label_with_bg(img, x1, y1, text, bg_color,
                        font_scale=0.9, text_thickness=2, bg_alpha=0.65):
    """
    Draws a filled, alpha-blended rectangle behind the label text for readability.
    Modifies `img` in-place and returns it.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, font_scale, text_thickness)

    pad = 4
    x0 = int(x1)
    y0 = int(max(0, y1 - th - base - 2*pad))
    x2 = int(min(img.shape[1]-1, x0 + tw + 2*pad))
    y2 = int(y1)

    # draw bg with alpha
    roi = img[y0:y2, x0:x2].copy()
    bg = roi.copy()
    cv2.rectangle(bg, (0, 0), (x2-x0, y2-y0), bg_color, -1)
    blended = (bg_alpha * bg + (1 - bg_alpha) * roi).astype(np.uint8)
    img[y0:y2, x0:x2] = blended

    # choose black/white text for contrast
    # (luminance in BGR because OpenCV uses BGR)
    b, g, r = bg_color
    lum = 0.114*b + 0.587*g + 0.299*r
    txt_color = (0, 0, 0) if lum > 150 else (255, 255, 255)

    # draw text
    cv2.putText(img, text, (x0 + pad, y2 - base - pad),
                font, font_scale, txt_color, text_thickness, cv2.LINE_AA)
    return img

import os
import torch
import matplotlib.pyplot as plt

@torch.inference_mode()
def visualize_all_val(
    model,
    val_ds,
    device,
    save_dir: str = "viz/val_all",
    score_thresh: float = 0.5,
    alpha: float = 0.35,
    batch_size: int = 4,
    start: int = 0,
    max_images: int | None = None,
    # make labels pop (these are the args your new draw_on_image supports)
    box_thickness: int = 4,
    font_scale: float = 1.1,
    text_thickness: int = 3,
    text_bg_alpha: float = 0.8,
):
    """
    Runs inference on the entire validation dataset (or a slice) and saves visualizations.

    Requires your `draw_on_image(img_t, out, id_to_name, score_thresh, alpha, ...)`
    and a mapping dict `ID_TO_NAME` in scope.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    total = len(val_ds)
    if max_images is None:
        end = total
    else:
        end = min(total, start + max_images)

    indices = list(range(start, end))
    if not indices:
        print("[viz] nothing to visualize (empty range).")
        return

    print(f"[viz] visualizing {len(indices)} images → {save_dir}")
    for s in range(0, len(indices), batch_size):
        chunk = indices[s:s + batch_size]

        # gather a mini-batch
        imgs_gpu, imgs_cpu = [], []
        for idx in chunk:
            img_t, _ = val_ds[idx]           # img_t: (3,H,W) in [0,1]
            imgs_cpu.append(img_t)           # keep CPU copy for drawing
            imgs_gpu.append(img_t.to(device))

        # forward pass (list[dict] of boxes/labels/scores/masks)
        outs = model(imgs_gpu)

        # draw & save each image in the chunk
        for j, (img_t, out) in enumerate(zip(imgs_cpu, outs)):
            # (N,1,H,W) -> (N,H,W) if needed
            if "masks" in out and torch.is_tensor(out["masks"]) and out["masks"].dim() == 4 and out["masks"].shape[1] == 1:
                out["masks"] = out["masks"].squeeze(1)

            plt.figure(figsize=(7, 7))
            draw_on_image(
                img_t, out, ID_TO_NAME,
                score_thresh=score_thresh, alpha=alpha,
                box_thickness=box_thickness,
                font_scale=font_scale,
                text_thickness=text_thickness,
                text_bg_alpha=text_bg_alpha,
            )
            out_path = os.path.join(save_dir, f"val_{chunk[j]:05d}.png")
            plt.savefig(out_path, bbox_inches="tight", dpi=120)
            plt.close()
            print(f"[viz] saved {out_path}")

def _draw_label_with_bg(img, x1, y1, text, bg_color,
                        font_scale=0.9, text_thickness=2, bg_alpha=0.65):
    """
    Draw a semi-transparent background behind `text` at left-top corner (x1, y1).
    Handles edge cases near borders to avoid OpenCV errors.
    Modifies `img` in-place (expects uint8 BGR).
    """
    H, W = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, font_scale, text_thickness)

    pad = 4
    # Anchor is above the box: banner spans [y0:y2, x0:x2]
    x0 = int(max(0, min(x1, W - 1)))
    y2 = int(max(0, min(y1, H - 1)))
    x2 = int(min(W, x0 + tw + 2 * pad))
    y0 = int(max(0, y2 - th - base - 2 * pad))

    # If the banner would be empty (touching top edge), try placing it *inside* the box area
    if y2 <= y0 or x2 <= x0:
        # place below the top edge instead
        y0 = int(min(H - 1, y2 + 1))
        y2 = int(min(H, y0 + th + base + 2 * pad))
        # Still empty? Give up on banner and just draw text.
        if y2 <= y0 or x2 <= x0:
            # choose contrasting text color
            b, g, r = bg_color
            lum = 0.114*b + 0.587*g + 0.299*r
            txt_color = (0, 0, 0) if lum > 150 else (255, 255, 255)
            cv2.putText(img, text, (x0, max(0, y1 - 2)), font, font_scale, txt_color, text_thickness, cv2.LINE_AA)
            return img

    roi = img[y0:y2, x0:x2]
    if roi.size == 0:  # extra guard
        b, g, r = bg_color
        lum = 0.114*b + 0.587*g + 0.299*r
        txt_color = (0, 0, 0) if lum > 150 else (255, 255, 255)
        cv2.putText(img, text, (x0, max(0, y1 - 2)), font, font_scale, txt_color, text_thickness, cv2.LINE_AA)
        return img

    # Create filled banner and alpha-blend onto ROI
    banner = np.empty_like(roi)
    banner[:] = bg_color
    cv2.addWeighted(banner, bg_alpha, roi, 1.0 - bg_alpha, 0.0, dst=roi)

    # Text color for contrast
    b, g, r = bg_color
    lum = 0.114*b + 0.587*g + 0.299*r
    txt_color = (0, 0, 0) if lum > 150 else (255, 255, 255)

    # Draw text inside the banner with padding
    cv2.putText(img, text, (x0 + pad, y2 - base - pad),
                font, font_scale, txt_color, text_thickness, cv2.LINE_AA)
    return img


if __name__ == '__main__':

    ds = CVATPolygonDataset("../dataset/combined_core.xml", "../dataset/images")
    print("Number of images in dataset:", len(ds))
    
    
    xml_root = Path("../dataset/images")  # your image_root
    missing = []
    for rec in ds.items:  # ds is your CVATPolygonDataset instance
        raw = rec['name']
        # what you currently try:
        p1 = (xml_root / raw)
        # common alternative: strip any subdirs and try just the basename
        p2 = xml_root / Path(raw).name
        if not p1.exists() and not p2.exists():
            missing.append(raw)
    
    indices = np.arange(len(ds))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)
    
    
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)
    BATCH_SIZE = 12  # paper uses 5
    NUM_WORKERS = 2
    
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn, persistent_workers=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 1 + 10  # background + your 10 labels in LABEL_MAP
    
    model = build_slit_maskrcnn(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, nesterov=True, weight_decay=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    
    EPOCHS = 20
    
    
    def train_one_epoch(model, loader, optimizer, scaler, device):
        model.train()
        running = defaultdict(float)
        n = 0
        for images, targets in loader:
            images = [im.to(device) for im in images]
            targets = [{k: (v.to(device) if hasattr(v, "to") else v) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                losses = model(images, targets)  # dict of losses from torchvision
                loss = sum(losses.values())
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            n += 1
            for k, v in losses.items():
                running[k] += float(v.detach().cpu())
        return {k: v / max(n, 1) for k, v in running.items()}
    
    
    @torch.inference_mode()
    def validate_one_epoch(model, loader, device):
        running = defaultdict(float)
        n = 0
        was_training = model.training
        
        # Force train() to make the model return losses
        model.train()
        for images, targets in loader:
            images = [im.to(device) for im in images]
            targets = [{k: (v.to(device) if hasattr(v, "to") else v) for k, v in t.items()} for t in targets]
            
            losses = model(images, targets)  # dict of tensors
            # sum of all losses for convenience
            total = sum(losses.values())
            
            n += 1
            running["total_loss"] += float(total.detach().cpu())
            for k, v in losses.items():
                running[k] += float(v.detach().cpu())
        
        # restore previous mode
        model.train(was_training)
        
        return {k: v / max(n, 1) for k, v in running.items()}
    
    '''
    best_val = float("inf")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_losses = train_one_epoch(model, train_loader, optimizer, scaler, device)
        val_losses = validate_one_epoch(model, val_loader, device)
        val_sum = sum(val_losses.values())
        print(f"Epoch {epoch:03d} | "
              f"train: {sum(train_losses.values()):.4f} "
              f"(rpn_cls {train_losses.get('loss_objectness', 0):.3f}, "
              f"rpn_box {train_losses.get('loss_rpn_box_reg', 0):.3f}, "
              f"roi_cls {train_losses.get('loss_classifier', 0):.3f}, "
              f"roi_box {train_losses.get('loss_box_reg', 0):.3f}, "
              f"mask {train_losses.get('loss_mask', 0):.3f}) | "
              f"val: {val_sum:.4f} | {time.time() - t0:.1f}s")
        
        # Save best-by-val-loss checkpoint
        if val_sum < best_val:
            best_val = val_sum
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch},
                       f"slitnet_best.pt")
    '''
    # preview_predictions(model, val_loader, device, N=6, score_thresh=0.5, alpha=0.4)
    

    ckpt = torch.load("slitnet_best.pt", map_location="cpu")
    
    NUM_CLASSES = ckpt.get("num_classes", 11)  # fallback to what you trained originally
    model = build_slit_maskrcnn(num_classes=NUM_CLASSES)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    model.eval()
    print(f"Loaded epoch {ckpt.get('epoch', '?')} with NUM_CLASSES={NUM_CLASSES}")

    ID_TO_NAME = {v: k for k, v in LABEL_MAP.items()}

    model.eval()
    '''
    matplotlib.use("Agg")  # safe on all machines
    
    outdir = Path("viz/latest")
    outdir.mkdir(parents=True, exist_ok=True)
    
    images, outs = predict_batch(model, val_loader, device, score_thresh=0.5, max_batches=1)
    
    print(f"batch images: {len(images)}, outs: {len(outs)}")
    for i in range(min(len(images), 6)):
        # tiny debug print so you know if there are detections
        b = outs[i].get("boxes")
        s = outs[i].get("scores")
        print(f"[img {i}] #boxes={0 if b is None else b.shape[0]}",
              f"#scores={0 if s is None else s.shape[0]}")
        
        plt.figure(figsize=(6, 6))
        draw_on_image(images[i], outs[i], ID_TO_NAME, score_thresh=None, alpha=0.35)
        plt.tight_layout()
        # save to disk; works even if there's no GUI
        plt.savefig(outdir / f"pred_{i:03d}.png", dpi=150, bbox_inches="tight", pad_inches=0)
        plt.close()
    
    print(f"Saved previews to {outdir.resolve()}")
    
    
    visualize_all_val(
        model, val_ds, device,
        save_dir="viz/val_all",
        score_thresh=0.5, alpha=0.35,
        batch_size=4,       # adjust for your VRAM
        start=0,            # where to start in val set
        max_images=None     # or a number to limit
    )
    '''
    print(LABEL_MAP)
    id2name = dict()
    for key, value in LABEL_MAP.items():
        id2name[value] = key
    preview_gt_only(model, val_loader, device,id2name, save_dir="viz/gt", max_images=60)
    print(type(val_ds))




  


