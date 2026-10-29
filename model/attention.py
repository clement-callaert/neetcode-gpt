import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.k = nn.Linear(embedding_dim, attention_dim, bias = False)
        self.q = nn.Linear(embedding_dim, attention_dim, bias = False)
        self.v = nn.Linear(embedding_dim, attention_dim, bias = False)
        self.attention_dim = attention_dim
        self.embedding_dim = embedding_dim

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        K = self.k(embedded)
        Q = self.q(embedded)
        V = self.v(embedded)
        score =  (Q @ torch.transpose(K, 1, 2)) / self.attention_dim ** 0.5
        M = torch.tril(torch.ones((K.shape[1], K.shape[1])))
        mask = M == 0
        score = score.masked_fill(mask, float('-inf'))
        score = nn.functional.softmax(score, dim=2)
        return torch.round(score @ V, decimals=4)



