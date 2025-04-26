import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        # TODO: Implement forward pass
        # Compute the softmax in a numerically stable way
        # Apply it to the dimension specified by the `dim` parameter
        # refer writeup for equations
        
        Z_shifted = Z - np.max(Z, axis=self.dim, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        sum_exp_Z = np.sum(exp_Z, axis=self.dim, keepdims=True)
        self.A = exp_Z / sum_exp_Z
        
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # TODO: Implement backward pass

        # Get the shape of the input
        shape = self.A.shape
        # Find the dimension along which softmax was applied
        if self.dim < 0:
            dim = len(shape) + self.dim
        else:
            dim = self.dim
        C = shape[dim]

        # Reshape input to 2D
        if len(shape) > 2:
            A_trans = np.moveaxis(self.A, dim, -1)
            dLdA_trans = np.moveaxis(dLdA, dim, -1)
            flat_shape = (-1, C)
            A_flat = A_trans.reshape(flat_shape)
            dLdA_flat = dLdA_trans.reshape(flat_shape)
        else:
            A_flat = self.A
            dLdA_flat = dLdA
            flat_shape = shape

        N, _ = A_flat.shape
        dLdZ_flat = np.zeros_like(A_flat)
        for i in range(N):
            # Jacobian J = diag(a_i) - a_i outer a_i
            J = np.diag(A_flat[i]) - np.outer(A_flat[i], A_flat[i])
            # Chain rule: dL/dZ_i = dL/dA_i · J
            dLdZ_flat[i] = np.dot(dLdA_flat[i], J)

        # Reshape back to original dimensions if necessary
        if len(shape) > 2:
            dLdZ_trans = dLdZ_flat.reshape(A_trans.shape)
            dLdZ = np.moveaxis(dLdZ_trans, -1, dim)
        else:
            dLdZ = dLdZ_flat

        return dLdZ