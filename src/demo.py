"""Live demo: python src/demo.py --image some.dcm|png --out result.png"""
import argparse, cv2, numpy as np, pydicom, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataset import IMG, lung_mask
from model import PneumoCAMNet
from cam import compute_cam, cam_to_boxes

ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True); ap.add_argument("--ckpt", default="best.pt")
ap.add_argument("--out", default="result.png"); ap.add_argument("--thr", type=float, default=0.45)
a = ap.parse_args()
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
model = PneumoCAMNet(); model.load_state_dict(ck["model"]); model.eval()
raw = (pydicom.dcmread(a.image).pixel_array if a.image.endswith(".dcm")
       else cv2.imread(a.image, 0)).astype(np.float32)
raw = cv2.resize(raw, (IMG, IMG), interpolation=cv2.INTER_AREA)
seg = raw * lung_mask(raw)
x = np.stack([(raw - ck["mean"][0]) / ck["std"][0], (seg - ck["mean"][1]) / ck["std"][1]]).astype(np.float32)
p, cam = compute_cam(model, torch.from_numpy(x)[None])
fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
ax[0].imshow(raw, cmap="gray"); ax[0].set_title("Input"); ax[1].imshow(raw, cmap="gray")
ax[1].set_title(f"P(pneumonia)={p[0]:.2f}")
if p[0] > 0.5:
    ax[1].imshow(cam[0], cmap="jet", alpha=0.4)
    for x1, y1, x2, y2 in cam_to_boxes(cam[0], a.thr):
        ax[1].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="cyan", lw=2))
for t in ax: t.axis("off")
plt.tight_layout(); plt.savefig(a.out, dpi=150); print("saved", a.out, "| P =", round(float(p[0]), 3))
