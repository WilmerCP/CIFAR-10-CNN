# modelo.py
import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_clases=10):
        super().__init__()
        self.features = nn.Sequential(
            # Bloque 1: [3, 32, 32] -> [32, 16, 16]
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Bloque 2: [32, 16, 16] -> [64, 8, 8]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Bloque 3: [64, 8, 8] -> [128, 4, 4]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                 # [128, 4, 4] -> [2048]
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_clases),   # 10 salidas (logits)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


if __name__ == "__main__":
    model = CNN()
    x = torch.randn(64, 3, 32, 32)
    print(model(x).shape)
    print(sum(p.numel() for p in model.parameters()), "parámetros")