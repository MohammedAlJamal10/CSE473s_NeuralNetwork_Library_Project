import numpy as np

class MSELoss:
    """
    Mean Squared Error: L = (1/N) * sum((y_true - y_pred)^2)
    """
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_true - y_pred) ** 2)

    def backward(self):
        # dL/dy_pred = 2 * (y_pred - y_true) / N
        n = self.y_true.size
        return (2.0 / n) * (self.y_pred - self.y_true)
