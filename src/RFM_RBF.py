import numpy as np

class RFM_RBF():
    """
    rbf(gaussian) kernel of GPy k(x, y) = variance * exp(- 0.5 * ||x - y||_2^2 / lengthscale**2)
    """
    def __init__(self, lengthscales, input_dim, variance=1.0, basis_dim=1000):
        self.basis_dim = basis_dim
        self.std = np.sqrt(variance)
        self.random_weights = (1 / lengthscales) * \
            np.random.normal(size=(basis_dim, input_dim))
        self.random_offset = np.random.uniform(0, 2 * np.pi, size=basis_dim)

    def transform(self, X):
        X_transform = X.dot(self.random_weights.T) + self.random_offset
        X_transform = self.std * np.sqrt(2 / self.basis_dim) * np.cos(X_transform)
        return X_transform