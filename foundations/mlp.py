import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        n = len(weights)
        z = x
        a = 0
        for i in range(n-1):
            a = z @ weights[i] + biases[i]
            z = np.maximum(0, a)
        z = z @ weights[-1] + biases[-1]
        return z
        
