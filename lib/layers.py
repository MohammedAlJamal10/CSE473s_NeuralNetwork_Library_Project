import numpy as np

class Layer:
    """
    Abstract base class for layers.
    Each layer must implement forward() and backward().
    """
    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def parameters(self):
        # Return list of (param, grad) pairs
        return []


class Dense(Layer):
    """
    Fully connected layer: y = xW + b
    inputs: (batch_size, input_dim)
    outputs: (batch_size, output_dim)
    """
    def __init__(self, input_dim, output_dim):
        limit = np.sqrt(6.0 / (input_dim + output_dim))
        self.W = np.random.uniform(-limit, limit, (input_dim, output_dim))
        self.b = np.zeros((1, output_dim), dtype=np.float64)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs  # (batch_size, input_dim)
        return inputs @ self.W + self.b  # (batch_size, output_dim)

    def backward(self, grad_output):
        """
        grad_output is dL/dY, already averaged by the loss.
        DO NOT divide by batch size again here.
        """
        # Gradients w.r.t parameters
        dW = self.inputs.T @ grad_output          # (input_dim, output_dim)
        db = np.sum(grad_output, axis=0, keepdims=True)  # (1, output_dim)

        # Update gradient buffers IN PLACE so optimizer sees them
        self.dW[...] = dW
        self.db[...] = db


        # Gradient w.r.t. inputs
        grad_input = grad_output @ self.W.T            # (batch_size, input_dim)
        return grad_input

    def parameters(self):
        return [
            (self.W, self.dW),
            (self.b, self.db)
        ]
