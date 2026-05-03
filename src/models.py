import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """
    Baseline model for course requirement:
    - >= 3 conv layers
    - pooling included
    - no BatchNorm / no Dropout
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 96 -> 48
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 48 -> 24
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 24 -> 12
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class ImprovedCNN(nn.Module):
    """
    Improved model for comparison:
    - BatchNorm + Dropout
    - deeper channels
    - AdaptiveAvgPool to reduce FC parameters
    """

    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 96 -> 48
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 48 -> 24
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 24 -> 12
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class ConfigurableCNN(nn.Module):
    """
    Configurable model used by ablation experiments.
    Keeps same depth as baseline while exposing:
    - BatchNorm on/off
    - Dropout value
    """

    def __init__(self, num_classes=10, use_bn=False, dropout=0.0):
        super().__init__()

        def maybe_bn(channels):
            return nn.BatchNorm2d(channels) if use_bn else nn.Identity()

        dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            maybe_bn(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 96 -> 48
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            maybe_bn(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 48 -> 24
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            maybe_bn(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 24 -> 12
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 256),
            nn.ReLU(inplace=True),
            dropout_layer,
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_model(model_name: str, num_classes=10, **kwargs):
    name = model_name.lower()
    if name == "baseline":
        return BaselineCNN(num_classes=num_classes)
    if name == "improved":
        return ImprovedCNN(num_classes=num_classes)
    if name == "configurable":
        return ConfigurableCNN(
            num_classes=num_classes,
            use_bn=bool(kwargs.get("use_bn", False)),
            dropout=float(kwargs.get("dropout", 0.0)),
        )
    raise ValueError(f"Unsupported model_name: {model_name}")
