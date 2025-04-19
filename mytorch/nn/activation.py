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
        
        # Subtract max for numerical stability
        Z_shifted = Z - np.max(Z, axis=self.dim, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        # Sum along the specified dimension
        sum_exp_Z = np.sum(exp_Z, axis=self.dim, keepdims=True)
        # Compute softmax
        self.A = exp_Z / sum_exp_Z
        
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # For the special case when input is 2D and dim=-1 (last dimension)
        if len(self.A.shape) == 2 and self.dim == -1:
            N, C = self.A.shape
            dLdZ = np.zeros_like(self.A)
            
            for i in range(N):
                # Create the Jacobian matrix for this example
                J = np.diag(self.A[i]) - np.outer(self.A[i], self.A[i])
                # Apply chain rule: dL/dZ = dL/dA * dA/dZ
                dLdZ[i] = np.dot(dLdA[i], J)
                
            return dLdZ
        
        # For the general case
        dLdZ = np.zeros_like(dLdA)
        
        # Handle arbitrary dimensions by looping through all elements
        # except along the softmax dimension
        if self.dim < 0:
            dim = len(self.A.shape) + self.dim
        else:
            dim = self.dim
        
        # Get iterator over all indices except along softmax dimension
        indices = np.ndindex(*(self.A.shape[:dim] + self.A.shape[dim+1:]))
        
        for idx in indices:
            # Insert softmax dimension to get full index
            full_idx = idx[:dim] + (slice(None),) + idx[dim:]
            
            # Get softmax outputs and gradients for this slice
            A_slice = self.A[full_idx]
            dLdA_slice = dLdA[full_idx]
            
            # Create Jacobian matrix for this slice
            J = np.diag(A_slice) - np.outer(A_slice, A_slice)
            
            # Apply chain rule
            dLdZ[full_idx] = np.dot(dLdA_slice, J)
        
        return dLdZ
 

    