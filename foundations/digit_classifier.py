import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid
        self.layer_1 = nn.Linear(784, 512)
        self.dropout = nn.Dropout(0.2)
        self.layer_2 = nn.Linear(512, 10)

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        # images shape: (batch_size, 784)
        # Return the model's prediction to 4 decimal places
        x = images
        x = self.layer_1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.layer_2(x)
        x = nn.functional.sigmoid(x)
        return torch.round(x, decimals=4)
