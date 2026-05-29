import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden=(256, 128, 64), dropout=0.3):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_model(weights_path: str, input_dim: int) -> MLP:
    model = MLP(input_dim)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model
