import torch
import torch.nn as nn
import torch.nn.functional as F


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


class ImprovedLongCNN(nn.Module):
    """
    Deeper backbone than the 3-pool Configurable path: six spatial reductions
    (five MaxPool2d(2): 96→48→24→12→6→3, then AdaptiveAvgPool2d(1,1)).
    Conv-BN-ReLU blocks; classifier uses Dropout like ImprovedCNN.
    """

    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 96 -> 48
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 48 -> 24
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 24 -> 12
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 12 -> 6
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 6 -> 3
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


class ImprovedLongerCNN(nn.Module):
    """
    Nine Conv2d layers (vs six in ImprovedLongCNN), same training recipe target as exp6:
    extra 128-channel convs at 24x24 and 12x12 before pooling; then 128->256 tail + global pool.
    Spatial: 96→48→24 (extra conv) →12 (three convs) →6→3→1.
    """

    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.conv9 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x)), inplace=True))
        x = self.pool(F.relu(self.bn2(self.conv2(x)), inplace=True))
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = self.pool(F.relu(self.bn4(self.conv4(x)), inplace=True))
        x = F.relu(self.bn5(self.conv5(x)), inplace=True)
        x = F.relu(self.bn6(self.conv6(x)), inplace=True)
        x = self.pool(F.relu(self.bn7(self.conv7(x)), inplace=True))
        x = self.pool(F.relu(self.bn8(self.conv8(x)), inplace=True))
        x = self.gap(F.relu(self.bn9(self.conv9(x)), inplace=True))
        return self.classifier(x)


class ImprovedLonger12CNN(nn.Module):
    """
    Twelve Conv2d layers: same downsampling schedule as ImprovedLongerCNN, with three extra
    128-channel 3x3 blocks — one at 24x24 after conv3, two at 12x12 after conv6 — before conv7.
    Spatial: 96→48→24 (conv3, conv3b) →12 (conv4–conv6c) →6→3→1.
    """

    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3b = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv6b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6b = nn.BatchNorm2d(128)
        self.conv6c = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6c = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.conv9 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x)), inplace=True))
        x = self.pool(F.relu(self.bn2(self.conv2(x)), inplace=True))
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = F.relu(self.bn3b(self.conv3b(x)), inplace=True)
        x = self.pool(F.relu(self.bn4(self.conv4(x)), inplace=True))
        x = F.relu(self.bn5(self.conv5(x)), inplace=True)
        x = F.relu(self.bn6(self.conv6(x)), inplace=True)
        x = F.relu(self.bn6b(self.conv6b(x)), inplace=True)
        x = F.relu(self.bn6c(self.conv6c(x)), inplace=True)
        x = self.pool(F.relu(self.bn7(self.conv7(x)), inplace=True))
        x = self.pool(F.relu(self.bn8(self.conv8(x)), inplace=True))
        x = self.gap(F.relu(self.bn9(self.conv9(x)), inplace=True))
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


class ConfigurableCNN15(nn.Module):
    """
    Exp2-style recipe (aug + optional BN + same classifier as ConfigurableCNN) with 15 conv layers:
    first three match ConfigurableCNN (32→64→128 with pool after each), then twelve 128→128×3×3
    blocks at 12×12 (no extra pooling), then Flatten + Linear head.
    """

    def __init__(self, num_classes=10, use_bn=False, dropout=0.0):
        super().__init__()

        def maybe_bn(channels):
            return nn.BatchNorm2d(channels) if use_bn else nn.Identity()

        dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        layers = [
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
        ]
        for _ in range(12):
            layers.extend(
                [
                    nn.Conv2d(128, 128, kernel_size=3, padding=1),
                    maybe_bn(128),
                    nn.ReLU(inplace=True),
                ]
            )
        self.features = nn.Sequential(*layers)
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
    if name == "improved_long":
        return ImprovedLongCNN(
            num_classes=num_classes,
            dropout=float(kwargs.get("dropout", 0.3)),
        )
    if name == "improved_longer":
        return ImprovedLongerCNN(
            num_classes=num_classes,
            dropout=float(kwargs.get("dropout", 0.3)),
        )
    if name in ("improved_longer_12", "improved_longer12"):
        return ImprovedLonger12CNN(
            num_classes=num_classes,
            dropout=float(kwargs.get("dropout", 0.3)),
        )
    if name == "configurable":
        return ConfigurableCNN(
            num_classes=num_classes,
            use_bn=bool(kwargs.get("use_bn", False)),
            dropout=float(kwargs.get("dropout", 0.0)),
        )
    if name in ("configurable_15conv", "configurable15"):
        return ConfigurableCNN15(
            num_classes=num_classes,
            use_bn=bool(kwargs.get("use_bn", False)),
            dropout=float(kwargs.get("dropout", 0.0)),
        )
    raise ValueError(f"Unsupported model_name: {model_name}")
