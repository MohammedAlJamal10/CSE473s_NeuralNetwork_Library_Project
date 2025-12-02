import numpy as np
from .layers import Layer

class ReLU(Layer):
    def __init__(self):
        self.mask = None

    def forward(self, inputs):
        self.mask = inputs > 0
        return np.maximum(0, inputs)

    def backward(self, grad_output):
        return grad_output * self.mask.astype(grad_output.dtype)


class Sigmoid(Layer):
    def __init__(self):
        self.out = None

    def forward(self, inputs):
        # sigmoid(x) = 1 / (1 + exp(-x))
        self.out = 1.0 / (1.0 + np.exp(-inputs))
        return self.out

    def backward(self, grad_output):
        # derivative: s * (1 - s)
        return grad_output * self.out * (1.0 - self.out)


class Tanh(Layer):
    def __init__(self):
        self.out = None

    def forward(self, inputs):
        self.out = np.tanh(inputs)
        return self.out

    def backward(self, grad_output):
        # derivative: 1 - tanh^2(x)
        return grad_output * (1.0 - self.out ** 2)


class Softmax(Layer):
    def __init__(self):
        self.out = None

    def forward(self, inputs):
        # numerically stable softmax
        shifted = inputs - np.max(inputs, axis=1, keepdims=True)
        exp = np.exp(shifted)
        self.out = exp / np.sum(exp, axis=1, keepdims=True)
        return self.out

    def backward(self, grad_output):
        """
        General softmax backward (for MSE or custom losses).
        For cross-entropy softmax simplifications you would usually
        do it inside the loss. But here we support general case.
        """
        batch_size, num_classes = self.out.shape
        grad_input = np.zeros_like(grad_output)

        # Slow but clear implementation
        for i in range(batch_size):
            y = self.out[i].reshape(-1, 1)
            # Jacobian of softmax
            J = np.diagflat(y) - y @ y.T
            grad_input[i] = J @ grad_output[i]
        return grad_input
