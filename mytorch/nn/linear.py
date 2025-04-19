import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        # DO NOT MODIFY
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)


    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        # DO NOT MODIFY
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)
        
        Handles arbitrary batch dimensions like PyTorch
        """
        # TODO: Implement forward pass
        
        # Store input for backward pass
        self.A = A
        
        # Z = A @ W^T + b
        Z = np.matmul(A, self.W.T) + self.b
        
        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """

        # TODO: Implement backward pass
        # Compute gradients (refer to the equations in the writeup)
        # Compute gradients
        # dL/dA = dL/dZ @ W
        self.dLdA = np.matmul(dLdZ, self.W)
        
        # dL/dW = dL/dZ^T @ A
        # Reshape dLdZ to handle arbitrary batch dimensions
        dLdZ_reshaped = dLdZ.reshape(-1, dLdZ.shape[-1])
        A_reshaped = self.A.reshape(-1, self.A.shape[-1])
        self.dLdW = np.matmul(dLdZ_reshaped.T, A_reshaped)
        
        # dL/db = sum of dL/dZ across all dimensions except the last
        self.dLdb = np.sum(dLdZ, axis=tuple(range(dLdZ.ndim - 1)))
        
        # Return gradient of loss wrt input
        return self.dLdA
