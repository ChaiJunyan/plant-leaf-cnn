"""Classic ResNet-18 and controlled depthwise/SE variants (random init only)."""
import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, ResNet


class SE(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, max(channels // reduction, 1), 1)
        self.fc2 = nn.Conv2d(max(channels // reduction, 1), channels, 1)

    def forward(self, x):
        gate = torch.sigmoid(self.fc2(torch.relu(self.fc1(self.pool(x)))))
        return x * gate


class SEBasicBlock(BasicBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.se = SE(self.bn2.num_features)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.se(self.bn2(self.conv2(out)))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


def separable(conv):
    """Linear DW+PW factorization; keep original external BN and activation."""
    assert conv.kernel_size == (3, 3) and conv.groups == 1 and conv.bias is None
    module = nn.Sequential(
        nn.Conv2d(conv.in_channels, conv.in_channels, 3, stride=conv.stride,
                  padding=conv.padding, dilation=conv.dilation,
                  groups=conv.in_channels, bias=False),
        nn.Conv2d(conv.in_channels, conv.out_channels, 1, bias=False),
    )
    for layer in module:
        nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
    return module


VARIANTS = ("resnet18", "resnet18_ds", "resnet18_se", "resnet18_ds_se")


def build_model(name="resnet18", num_classes=38):
    if name not in VARIANTS:
        raise ValueError(f"Unknown architecture: {name}")
    model = ResNet(SEBasicBlock if name.endswith("_se") else BasicBlock,
                   [2, 2, 2, 2], num_classes=num_classes)
    if "_ds" in name:
        for stage in (model.layer3, model.layer4):
            for block in stage:
                block.conv1 = separable(block.conv1)
                block.conv2 = separable(block.conv2)
    return model


@torch.no_grad()
def profile_model(model, image_size=128):
    """Count Conv2d/Linear MACs for batch=1. BN/pooling/SE gating excluded."""
    macs = 0
    handles = []

    def hook(layer, inputs, output):
        nonlocal macs
        if isinstance(layer, nn.Conv2d):
            macs += output.numel() * (layer.in_channels // layer.groups) * layer.kernel_size[0] * layer.kernel_size[1]
        elif isinstance(layer, nn.Linear):
            macs += output.numel() * layer.in_features

    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            handles.append(layer.register_forward_hook(hook))
    mode = model.training
    try:
        model.eval()
        parameter = next(model.parameters())
        output = model(torch.zeros(1, 3, image_size, image_size, device=parameter.device, dtype=parameter.dtype))
        assert output.shape == (1, model.fc.out_features)
    finally:
        for handle in handles:
            handle.remove()
        model.train(mode)
    return {"parameters": sum(p.numel() for p in model.parameters()),
            "conv_linear_macs": macs, "image_size": image_size,
            "mac_definition": "Conv2d+Linear, batch=1; BN, activations, pooling and elementwise gates excluded; 1 MAC = 2 arithmetic FLOPs"}
