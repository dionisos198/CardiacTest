
import numpy as np

class SigmoidBinaryCrossEntropyLoss:
    def __init__(self):
        self.loss = None  # Loss function
        self.y = None  # Sigmoid output
        self.t = None  # True labels (binary, 0 or 1)

    def forward(self, x, t):
        self.t = t  # True labels
        self.y = self.sigmoid(x)  # Apply sigmoid activation--->삭제후 추가
        self.loss = self.binary_cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def binary_cross_entropy_error(self, y, t):
        if y.ndim == 2:
            y = y.reshape(y.shape[0], -1)
            t = t.reshape(t.shape[0], -1)

        batch_size = y.shape[0]
        # Add a small epsilon to prevent log(0) issues
        epsilon = 1e-7
        return -np.sum(t * np.log(y + epsilon) + (1 - t) * np.log(1 - y + epsilon)) / batch_size
