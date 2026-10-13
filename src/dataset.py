"""Data loading for RSNA Pneumonia (Kaggle). Builds a 128x128 cache, splits 70/20/10."""
import cv2, numpy as np, pandas as pd, pydicom, torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

IMG = 128
ORIG = 1024


def lung_mask(img):
    """Cheap lung-region heuristic (Otsu on blurred image). Replace with a trained
    U-Net (as in the paper) for better results - see README 'Extensions'."""
    u8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    u8 = cv2.GaussianBlur(u8, (5, 5), 0)
    _, m = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return (m > 0).astype(np.float32)


def build_cache(data_dir, out="cache.npz"):
    """Reads DICOMs, keeps only 'Lung Opacity' (pneumonia) and 'Normal' (drops
    'No Lung Opacity / Not Normal', as in the paper), resizes to 128x128."""
    d = Path(data_dir)
    labels = pd.read_csv(d / "stage_2_train_labels.csv")
    info = pd.read_csv(d / "stage_2_detailed_class_info.csv").drop_duplicates("patientId")
    info = info[info["class"].isin(["Lung Opacity", "Normal"])]
    ids = info.patientId.tolist()
    y = (info["class"] == "Lung Opacity").astype(int).values
    boxes = {}
    for r in labels.dropna().itertuples():
        boxes.setdefault(r.patientId, []).append((r.x, r.y, r.x + r.width, r.y + r.height))
    s = IMG / ORIG
    X, bx = [], []
    for i in ids:
        a = pydicom.dcmread(d / "stage_2_train_images" / f"{i}.dcm").pixel_array.astype(np.float32)
        X.append(cv2.resize(a, (IMG, IMG), interpolation=cv2.INTER_AREA))
        bx.append([tuple(v * s for v in b) for b in boxes.get(i, [])])
    b_arr = np.empty(len(bx), dtype=object)
    for k, v in enumerate(bx):
        b_arr[k] = v
    np.savez_compressed(out, X=np.stack(X), y=y, ids=np.array(ids), boxes=b_arr)
    print(f"Cached {len(ids)} images ({y.sum()} pneumonia, {(1 - y).sum()} normal) -> {out}")


def get_splits(y, seed=42):
    idx = np.arange(len(y))
    trval, te = train_test_split(idx, test_size=0.10, stratify=y, random_state=seed)
    tr, va = train_test_split(trval, test_size=0.20 / 0.90, stratify=y[trval], random_state=seed)
    return tr, va, te


class XrayDataset(Dataset):
    """Returns a 2-channel tensor: [normalised original, normalised lung-segmented]."""
    def __init__(self, cache, idx, mean, std):
        self.X, self.y, self.idx, self.mean, self.std = cache["X"], cache["y"], idx, mean, std

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        raw = self.X[j]
        seg = raw * lung_mask(raw)
        x = np.stack([(raw - self.mean[0]) / self.std[0], (seg - self.mean[1]) / self.std[1]])
        return torch.from_numpy(x.astype(np.float32)), int(self.y[j]), j


def compute_norm(cache, train_idx):
    """Per-pixel mean/std over the training set, for both channels (paper's normalisation)."""
    raw = cache["X"][train_idx]
    seg = np.stack([r * lung_mask(r) for r in raw])
    eps = 1e-6
    return (np.stack([raw.mean(0), seg.mean(0)]), np.stack([raw.std(0) + eps, seg.std(0) + eps]))
