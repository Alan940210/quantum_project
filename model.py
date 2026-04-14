import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 1x28x28 -> 32x28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                             # -> 32x14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),# -> 64x14x14
            nn.ReLU(),
            nn.MaxPool2d(2)                              # -> 64x7x7
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x