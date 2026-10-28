import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        var = np.mean(np.square(x))
        eps = 1e-5
        x_hat = x / np.sqrt(var+ eps)
        out = gamma * x_hat
        return np.round(out, 4)
