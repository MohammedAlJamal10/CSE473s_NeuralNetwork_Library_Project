class SGD:
    """
    Stochastic Gradient Descent optimizer.
    """

    def __init__(self, parameters, lr=0.01):
        """
        parameters: list of (param_array, grad_array) tuples
        lr: learning rate
        """
        self.parameters = parameters
        self.lr = lr

    def step(self):
        """Update all parameters in-place."""
        for param, grad in self.parameters:
            param -= self.lr * grad

    def zero_grad(self):
        """Reset all gradients to zero."""
        for _, grad in self.parameters:
            grad[...] = 0.0
