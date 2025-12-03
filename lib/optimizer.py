import numpy as np

class SGD:
    """
    Stochastic Gradient Descent optimizer.

    Holds references to (param, grad) pairs and applies:
        param -= lr * grad

    Gradients are expected to be computed by calling
    loss.backward() and model.backward(grad_loss) BEFORE step().
    """

    def __init__(self, parameters, lr=0.01):
        """
        parameters: list of (param, grad) tuples
        lr: learning rate
        """
        self.parameters = parameters
        self.lr = lr

    def step(self):
        """Apply one SGD update to all parameters."""
        for param, grad in self.parameters:
            param -= self.lr * grad

    def zero_grad(self):
        """Reset all gradients to zero."""
        for _, grad in self.parameters:
            grad[...] = 0.0


def iterate_minibatches(X, y=None, batch_size=32, shuffle=True):
    """
    Yield mini-batches of data (and labels if given).

    X: array of shape (N, ...)
    y: None or array of shape (N, ...)
    batch_size: int
    shuffle: whether to shuffle samples before creating batches

    Yields:
        if y is None: X_batch
        else: (X_batch, y_batch)
    """
    N = X.shape[0]
    indices = np.arange(N)
    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, N, batch_size):
        end_idx = start_idx + batch_size
        batch_idx = indices[start_idx:end_idx]
        if y is None:
            yield X[batch_idx]
        else:
            yield X[batch_idx], y[batch_idx]
