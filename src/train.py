import argparse, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import XrayDataset, compute_norm, get_splits
from model import PneumoCAMNet


def run_epoch(model, dl, dev, opt=None):
    train = opt is not None
    model.train(train)
    tot, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for x, y, _ in tqdm(dl, leave=False):
            x, y = x.to(dev), y.to(dev)
            out = model(x)
            loss = nn.functional.cross_entropy(out, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * len(y); correct += (out.argmax(1) == y).sum().item(); tot += len(y)
    return loss_sum / tot, correct / tot


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache.npz")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cache = np.load(a.cache, allow_pickle=True)
    tr, va, te = get_splits(cache["y"])
    mean, std = compute_norm(cache, tr)
    mk = lambda idx, sh: DataLoader(XrayDataset(cache, idx, mean, std), batch_size=a.bs, shuffle=sh)
    dtr, dva = mk(tr, True), mk(va, False)
    model = PneumoCAMNet().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    best = 0
    for ep in range(1, a.epochs + 1):
        tl, ta = run_epoch(model, dtr, dev, opt)
        vl, va_acc = run_epoch(model, dva, dev)
        print(f"epoch {ep:02d} train acc {ta:.4f} | val loss {vl:.4f} acc {va_acc:.4f}")
        if va_acc > best:
            best = va_acc
            torch.save({"model": model.state_dict(), "mean": mean, "std": std}, "best.pt")
    print("best val acc", best)
