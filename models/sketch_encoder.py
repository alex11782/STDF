import random

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50


class MixStyle(nn.Module):
    def __init__(self, p=0.5, alpha=0.1, enabled=True):
        super().__init__()
        self.p = p
        self.enabled = enabled
        self.beta = torch.distributions.Beta(alpha, alpha)

    def forward(self, x):
        if not self.enabled or not self.training or random.random() > self.p:
            return x
        batch = x.size(0)
        perm = torch.randperm(batch, device=x.device)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + 1e-6).sqrt()
        lmda = self.beta.sample((batch, 1, 1, 1)).to(x.device)
        mu_mix = mu * lmda + mu[perm] * (1.0 - lmda)
        sig_mix = sig * lmda + sig[perm] * (1.0 - lmda)
        return ((x - mu) / sig) * sig_mix + mu_mix


class RobustSketchEncoder(nn.Module):
    def __init__(
        self,
        output_dim=512,
        backbone="resnet50",
        use_mixstyle=True,
        mixstyle_p=0.5,
        mixstyle_alpha=0.1,
    ):
        super().__init__()
        if backbone == "resnet50":
            base = resnet50(weights="DEFAULT")
            in_features = 2048
        elif backbone == "resnet18":
            base = resnet18(weights="DEFAULT")
            in_features = 512
        else:
            raise ValueError(f"unsupported sketch backbone: {backbone}")

        self.backbone_name = backbone
        self.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.mixstyle1 = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha, enabled=use_mixstyle)
        self.layer2 = base.layer2
        self.mixstyle2 = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha, enabled=use_mixstyle)
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = nn.Linear(in_features, output_dim)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.mixstyle1(self.layer1(x))
        x = self.mixstyle2(self.layer2(x))
        x = self.layer4(self.layer3(x))
        return self.fc(torch.flatten(self.avgpool(x), 1))
