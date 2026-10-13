import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Sequential):
    def __init__(self, input_channels, output_channels):
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.GroupNorm(8, output_channels),
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(8, output_channels),
            nn.ReLU(),
        )


class Encoder(nn.Module):
    def __init__(self, inputs=1, widths=(32, 64, 128, 256)):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(a, b) for a, b in zip((inputs,) + widths[:-1], widths)]
        )

    def forward(self, image):
        features = []
        for i, block in enumerate(self.blocks):
            image = block(F.avg_pool2d(image, 2) if i else image)
            features.append(image)
        return features


class Decoder(nn.Module):
    def __init__(self, context=0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Block(256 + context, 128),
                Block(128 + 128 + context, 64),
                Block(64 + 64 + context, 32),
                Block(32 + 32 + context, 32),
            ]
        )
        self.output = nn.Conv2d(32, 1, 1)

    def forward(self, features, context=None):
        value = features[-1]
        for i, block in enumerate(self.blocks):
            if i:
                target = features[-i - 1]
                value = torch.cat(
                    [
                        F.interpolate(
                            value,
                            size=target.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        ),
                        target,
                    ],
                    1,
                )
            if context is not None:
                value = torch.cat(
                    [
                        value,
                        F.interpolate(
                            context,
                            size=value.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        ),
                    ],
                    1,
                )
            value = block(value)
        return self.output(value)


def segmentation_loss(probability, target):
    probability = probability.clamp(1e-6, 1 - 1e-6)
    bce = F.binary_cross_entropy(probability, target)
    axes = tuple(range(1, target.ndim))
    dice = (2 * (probability * target).sum(axes) + 1e-6) / (
        probability.sum(axes) + target.sum(axes) + 1e-6
    )
    return bce + 1 - dice.mean()
