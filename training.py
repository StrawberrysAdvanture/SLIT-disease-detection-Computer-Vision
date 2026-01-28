from models.mask_rcnn import build_slit_maskrcnn, build_base_maskrcnn
CLASSES = [
  "__background__",
  "cornea", "pupil", "uppereyelid", "lowereyelid", "lightreflex",
  "pinguecula", "conjnevus", "pseudophakia", "cataract", "irisnevus", "pterygium",
]
label2id = {c:i for i,c in enumerate(CLASSES)}

import xml.etree.ElementTree as ET
import numpy as np
import cv2
import torch

DROPPED = {"aciol", "subconj heme", "pkp"}  # rare diseases



def parse_cvat_xml_all_labels(xml_path, label2id):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    data = {}

    for img in root.findall("image"):
        name = img.attrib["name"]
        w = int(img.attrib["width"])
        h = int(img.attrib["height"])

        masks, labels, boxes = [], [], []

        for ann in img:
            label = ann.attrib.get("label")
            if (label is None) or (label in DROPPED) or (label not in label2id):
                continue

            pts = np.array(
                [[float(x), float(y)] for x, y in (p.split(",") for p in ann.attrib["points"].split(";"))],
                dtype=np.int32
            )

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 1)

            ys, xs = np.where(mask)
            if len(xs) == 0:  # degenerate polygon
                continue

            xmin, xmax = xs.min(), xs.max()
            ymin, ymax = ys.min(), ys.max()

            masks.append(mask)
            labels.append(label2id[label])
            boxes.append([xmin, ymin, xmax, ymax])

        # Keep only if we have at least one remaining instance (anatomy or disease)
        if len(boxes) == 0:
            continue

        data[name] = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": torch.tensor(np.stack(masks), dtype=torch.uint8),
        }

    return data

from torch.utils.data import Dataset
from PIL import Image
import os

class EyeDiseaseDataset(Dataset):
    def __init__(self, img_dir, xml_path, label2id, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms
        self.label2id = label2id

        self.annotations = parse_cvat_xml_all_labels(xml_path, self.label2id)
        self.images = list(self.annotations.keys())

        # optional safety
        self.images = [n for n in self.images if os.path.exists(os.path.join(self.img_dir, n))]

        print("Loaded images:", len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        target = self.annotations[img_name]

        target = dict(target)  # avoid accidental mutation across epochs
        target["image_id"] = torch.tensor([idx])
        target["area"] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (target["boxes"][:, 2] - target["boxes"][:, 0])
        target["iscrowd"] = torch.zeros((len(target["labels"]),), dtype=torch.int64)

        if self.transforms:
            image = self.transforms(image)

        return image, target
    
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = EyeDiseaseDataset(
    img_dir="/home/jackie/Documents/UCLA/eyes/dataset/images",
    xml_path="/home/jackie/Documents/UCLA/eyes/dataset/combined_core_train.xml",
    label2id = label2id,
    transforms=transform
)

test_dataset = EyeDiseaseDataset(
    img_dir="/home/jackie/Documents/UCLA/eyes/dataset/images",
    xml_path="/home/jackie/Documents/UCLA/eyes/dataset/combined_core_test.xml",
    label2id=label2id,
    transforms=transform
)

from torch.utils.data import DataLoader, WeightedRandomSampler

DISEASE_NAMES = {"pinguecula","conjnevus","pseudophakia","cataract","irisnevus","pterygium"}
DISEASE_IDS = {label2id[n] for n in DISEASE_NAMES}

def make_sampler(dataset, disease_boost=5.0, anatomy_only_weight=1.0):
    weights = []
    for img_name in dataset.images:
        labels = dataset.annotations[img_name]["labels"]  # Tensor [N]
        labels = labels.tolist() if hasattr(labels, "tolist") else list(labels)
        has_disease = any(int(l) in DISEASE_IDS for l in labels)

        w = disease_boost if has_disease else anatomy_only_weight
        weights.append(w)

    weights = torch.as_tensor(weights, dtype=torch.double)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler

def collate_fn(batch):
    return tuple(zip(*batch))
sampler = make_sampler(train_dataset, disease_boost=5.0)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    sampler=sampler,
    num_workers=0,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    test_dataset,
    batch_size=3,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)

if __name__ == "__main__":
    import torch
    from torch.cuda.amp import autocast, GradScaler
    import torch, os
    import numpy as np
    from PIL import Image
    
    
    def validate_sample(img_dir, img_name, target):
        # image size from disk
        w, h = Image.open(os.path.join(img_dir, img_name)).size
        
        boxes = target["boxes"]
        masks = target["masks"]
        labels = target["labels"]
        
        # finite checks
        if not torch.isfinite(boxes).all():
            return "boxes contain NaN/Inf"
        if boxes.numel() and (boxes[:, 0] < 0).any() or (boxes[:, 1] < 0).any() or (boxes[:, 2] > w).any() or (
                boxes[:, 3] > h).any():
            return f"box out of bounds (img {w}x{h})"
        
        # degenerate boxes
        if boxes.numel():
            if (boxes[:, 2] <= boxes[:, 0]).any() or (boxes[:, 3] <= boxes[:, 1]).any():
                bad = ((boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1])).nonzero().flatten().tolist()
                return f"degenerate box at idx {bad}"
        
        # mask sanity
        if masks.dtype not in (torch.uint8, torch.bool):
            return f"mask dtype {masks.dtype} (expected uint8/bool)"
        if masks.numel():
            # each instance should have at least 1 pixel
            areas = masks.flatten(1).sum(1).cpu().numpy()
            if (areas <= 0).any():
                bad = np.where(areas <= 0)[0].tolist()
                return f"empty mask at idx {bad}"
        # label range
        if labels.numel() and (labels < 1).any():
            return "label < 1 found"
        
        return None
    
    
    # scan a handful until you hit the bad one
    for i in range(min(len(train_dataset), 400)):
        img, tgt = train_dataset[i]
        img_name = train_dataset.images[i]
        err = validate_sample(train_dataset.img_dir, img_name, tgt)
        if err:
            print("BAD:", img_name, err)
            break
    else:
        print("No obvious issues in first 400 samples.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = build_slit_maskrcnn(num_classes=len(CLASSES))  # make sure this is correct
    model = build_base_maskrcnn(num_classes=len(CLASSES))
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    scaler = GradScaler(enabled=(device.type == "cuda"))  # AMP
    
    
    def train_one_epoch(epoch):
        model.train()
        running = 0.0
        
        for step, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=(device.type == "cuda")):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running += loss.item()
            
            if step % 20 == 0:
                ld = {k: float(v.detach().cpu()) for k, v in loss_dict.items()}
                print(f"Epoch {epoch} Step {step}/{len(train_loader)} "
                      f"loss={loss.item():.4f} " +
                      " ".join([f"{k}={v:.3f}" for k, v in ld.items()]))
        
        return running / max(len(train_loader), 1)
    
    
    @torch.no_grad()
    def evaluate():
        model.eval()
        # quick smoke eval: just run forward to ensure no runtime issues
        for images, targets in test_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)  # no targets in eval
            # outputs is list of dicts with boxes/labels/scores/masks
            break
        print("Eval forward pass OK.")
    
    
    num_epochs = 10
    # This exists on torchvision RPNHead:
    print("rpn head cls_out:", model.rpn.head.cls_logits.out_channels)  # should equal A
    print("rpn head bbox_out:", model.rpn.head.bbox_pred.out_channels)  # should equal A*4
    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(epoch)
        print(f"Epoch {epoch} avg_loss={avg_loss:.4f}")
        lr_scheduler.step()
        evaluate()
    
    torch.save(model.state_dict(), "maskrcnn_eye_multilabel.pth")
    print("Saved checkpoint: maskrcnn_eye_multilabel.pth")