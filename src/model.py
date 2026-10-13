import torch, torch.nn as nn


class PneumoCAMNet(nn.Module):
    """10 conv layers (3x3, zero-padded, ReLU) -> Global Average Pool -> ONE FC layer.
    That structure is exactly what CAM requires."""
    CFG = [32, 32, "M", 64, 64, "M", 128, 128, "M", 256, 256, 256, 256]  # 10 convs

    def __init__(self, in_ch=2, n_classes=2):
        super().__init__()
        layers, c = [], in_ch
        for v in self.CFG:
            if v == "M":
                layers.append(nn.MaxPool2d(2))
            else:
                layers += [nn.Conv2d(c, v, 3, padding=1), nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                c = v
        self.features = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c, n_classes)

    def forward(self, x, return_features=False):
        f = self.features(x)
        logits = self.fc(self.gap(f).flatten(1))
        return (logits, f) if return_features else logits
