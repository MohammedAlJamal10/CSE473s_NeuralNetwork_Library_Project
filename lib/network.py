class Sequential:
    """
    Simple container for layers, similar to Keras.Sequential.
    It handles forward and backward passes over a list of layers.
    """

    def __init__(self, layers):
        """
        layers: list of layer instances (Dense, ReLU, etc.)
        """
        self.layers = layers

    def forward(self, x):
        """
        Forward pass through all layers.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output):
        """
        Backward pass through all layers (reverse order).
        grad_output: gradient of the loss w.r.t. network output
        """
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

    def parameters(self):
        """
        Collect all (param, grad) pairs from layers that have parameters().
        """
        params = []
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                params.extend(layer.parameters())
        return params
