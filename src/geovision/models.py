"""Model zoo: a from-scratch CNN baseline plus transfer-learning backbones.

Every model exposes the same contract:
  build_model(name, ...)  -> nn.Module producing logits [B, num_classes]
  head_module(model, name) -> the classifier head (trained alone during warmup)
  cam_layer(model, name)   -> last conv block, used by Grad-CAM
"""

import torch.nn as nn
from torchvision import models

SUPPORTED = ("baseline_cnn", "resnet50", "efficientnet_b0", "convnext_tiny")


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class BaselineCNN(nn.Module):
    """4-stage VGG-style CNN trained from scratch - the reference point for transfer learning."""

    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(conv_block(3, 32), conv_block(32, 64),
                                      conv_block(64, 128), conv_block(128, 256))
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                        nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(name, num_classes, pretrained=True, dropout=0.3):
    if name == "baseline_cnn":
        return BaselineCNN(num_classes, dropout)

    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, num_classes))
        return m

    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.classifier[1].in_features, num_classes))
        return m

    if name == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[2] = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.classifier[2].in_features, num_classes))
        return m

    raise ValueError(f"Unknown model '{name}'. Choose from {SUPPORTED}")


def head_module(model, name):
    return model.fc if name == "resnet50" else model.classifier


def cam_layer(model, name):
    if name == "resnet50":
        return model.layer4[-1]
    return model.features[-1]


def set_backbone_trainable(model, name, trainable):
    head_ids = {id(p) for p in head_module(model, name).parameters()}
    for p in model.parameters():
        if id(p) not in head_ids:
            p.requires_grad = trainable


def param_groups(model, name, lr, backbone_lr_mult):
    """Discriminative learning rates: the pretrained backbone moves slower than the new head."""
    head_params = list(head_module(model, name).parameters())
    head_ids = {id(p) for p in head_params}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    mult = 1.0 if name == "baseline_cnn" else backbone_lr_mult
    return [{"params": backbone, "lr": lr * mult}, {"params": head_params, "lr": lr}]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
