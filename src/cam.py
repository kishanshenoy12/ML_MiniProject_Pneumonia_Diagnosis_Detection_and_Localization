"""CAM -> heatmap -> DFS clustering -> bounding boxes -> IoU."""
import numpy as np, torch, torch.nn.functional as F


@torch.no_grad()
def compute_cam(model, x, cls=1):
    """M_c(x,y) = sum_k w_k^c f_k(x,y). Returns P(pneumonia) and CAM scaled to [0,1], HxW."""
    model.eval()
    logits, feats = model(x, return_features=True)
    w = model.fc.weight[cls]
    cam = F.relu(torch.einsum("c,bchw->bhw", w, feats))
    cam = F.interpolate(cam[:, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    mn, mx = cam.amin((1, 2), keepdim=True), cam.amax((1, 2), keepdim=True)
    cam = (cam - mn) / (mx - mn + 1e-8)
    return logits.softmax(1)[:, cls].cpu().numpy(), cam.cpu().numpy()


def cam_to_boxes(cam, thr=0.45, min_area=30, std_k=2.0):
    """Threshold heatmap, cluster non-zero pixels with an (iterative) DFS, box each cluster,
    keep boxes whose area is within std_k standard deviations of the mean (paper step 4)."""
    mask = cam >= thr
    H, W = mask.shape
    seen = np.zeros_like(mask)
    boxes = []
    for sy, sx in zip(*np.nonzero(mask)):
        if seen[sy, sx]:
            continue
        stack, ys, xs = [(sy, sx)], [], []
        seen[sy, sx] = True
        while stack:
            y, x = stack.pop()
            ys.append(y); xs.append(x)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if len(ys) >= min_area:
            boxes.append((min(xs), min(ys), max(xs) + 1, max(ys) + 1))
    if len(boxes) > 2:
        areas = np.array([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes], float)
        keep = np.abs(areas - areas.mean()) <= std_k * areas.std()
        boxes = [b for b, k in zip(boxes, keep) if k]
    return boxes


def iou(pred_boxes, gt_boxes, size=128):
    """IoU between the union of predicted boxes and the union of ground-truth boxes."""
    def raster(bs):
        m = np.zeros((size, size), bool)
        for x1, y1, x2, y2 in bs:
            m[int(max(y1, 0)):int(min(np.ceil(y2), size)), int(max(x1, 0)):int(min(np.ceil(x2), size))] = True
        return m
    p, g = raster(pred_boxes), raster(gt_boxes)
    u = (p | g).sum()
    return float((p & g).sum() / u) if u else 0.0
