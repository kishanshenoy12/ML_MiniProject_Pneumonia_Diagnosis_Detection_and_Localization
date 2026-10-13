"""Test-set accuracy, confusion matrix, and CAM localisation IoU."""
import argparse, numpy as np, torch
from sklearn.metrics import confusion_matrix
from dataset import XrayDataset, get_splits
from model import PneumoCAMNet
from cam import compute_cam, cam_to_boxes, iou

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="cache.npz"); ap.add_argument("--ckpt", default="best.pt")
ap.add_argument("--thr", type=float, default=0.45)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
cache = np.load(a.cache, allow_pickle=True)
ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
model = PneumoCAMNet().to(dev); model.load_state_dict(ck["model"]); model.eval()
_, _, te = get_splits(cache["y"])
ds = XrayDataset(cache, te, ck["mean"], ck["std"])
ys, ps, ious = [], [], []
for i in range(len(ds)):
    x, y, j = ds[i]
    p, cam = compute_cam(model, x[None].to(dev))
    ys.append(y); ps.append(int(p[0] > 0.5))
    if y == 1 and p[0] > 0.5:  # localise true positives only
        ious.append(iou(cam_to_boxes(cam[0], a.thr), cache["boxes"][j]))
ys, ps = np.array(ys), np.array(ps)
print("Test accuracy:", (ys == ps).mean())
print("Confusion matrix [[TN FP],[FN TP]]:\n", confusion_matrix(ys, ps))
print(f"Mean IoU over {len(ious)} localised images (thr={a.thr}): {np.mean(ious):.4f}")
