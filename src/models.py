"""ImageNet-pretrained backbones with a new 6-way head, plus helpers for
freezing the backbone and for discriminative (layer-wise) learning rates."""
import torch
import torch.nn as nn
from torchvision import models as tvm

from .config import NUM_CLASSES

ARCHS = ["resnet18", "resnet34", "resnet50", "efficientnet_b0", "convnext_tiny"]


def build_model(arch, num_classes=NUM_CLASSES, pretrained=True, dropout=0.2):
    if arch.startswith("resnet"):
        weights = {"resnet18": tvm.ResNet18_Weights, "resnet34": tvm.ResNet34_Weights,
                   "resnet50": tvm.ResNet50_Weights}[arch].DEFAULT if pretrained else None
        m = getattr(tvm, arch)(weights=weights)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, num_classes))
    elif arch == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif arch == "convnext_tiny":
        m = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    else:
        raise ValueError(f"unknown arch {arch}; choose from {ARCHS}")
    return m


def split_layers(model, arch):
    """Return (early, middle, head) lists of modules."""
    if arch.startswith("resnet"):
        early = [model.conv1, model.bn1, model.layer1, model.layer2]
        middle = [model.layer3, model.layer4]
        head = [model.fc]
    else:  # efficientnet / convnext both expose .features and .classifier
        feats = list(model.features.children())
        k = len(feats) // 2
        early, middle, head = feats[:k], feats[k:], [model.classifier]
    return early, middle, head


def _params(mods):
    return [p for m in mods for p in m.parameters()]


def set_backbone_trainable(model, arch, trainable):
    early, middle, _ = split_layers(model, arch)
    for p in _params(early + middle):
        p.requires_grad = trainable


def param_groups(model, arch, lr, discriminative=True):
    """Layer-wise learning rates: lr/9 early, lr/3 middle, lr for the new head.
    Early layers hold generic ImageNet edges/textures, so they should move the least."""
    early, middle, head = split_layers(model, arch)
    if not discriminative:
        return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}]
    return [
        {"params": _params(early), "lr": lr / 9},
        {"params": _params(middle), "lr": lr / 3},
        {"params": _params(head), "lr": lr},
    ]


def gradcam_layer(model, arch):
    if arch.startswith("resnet"):
        return model.layer4[-1]
    return model.features[-1]


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["arch"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt
