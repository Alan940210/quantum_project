import torch.nn as nn
import pennylane as qml

dev = qml.device("default.qubit", wires=8)
@qml.qnode(dev, interface="torch")
def circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(8))
    qml.StronglyEntanglingLayers(weights[0], wires=range(4))
    qml.StronglyEntanglingLayers(weights[1], wires=range(4, 8))

    return [qml.expval(qml.PauliZ(i)) for i in range(8)]

shapes = {"weights" : (2, 1, 4, 3)}

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.DC = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                             # -> 8 * 14 * 14

            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                             # -> 8 * 7 * 7

            nn.Flatten(),
            nn.Linear(8 * 7 * 7, 8),
        )

        self.qlayer = qml.qnn.TorchLayer(circuit, shapes)

        self.end = nn.Linear(8, 10)

    def forward(self, x):
        x = self.DC(x)
        x = self.qlayer(x)
        x = self.end(x)
        return x