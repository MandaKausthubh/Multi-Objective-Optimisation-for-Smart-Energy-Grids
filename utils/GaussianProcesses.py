import numpy as np
from utils.Kernels import MeanKernel, CovKernel

# -------- Gaussian Process --------
class GaussianProcess():
    def __init__(
            self,
            mean_kernel : MeanKernel,
            cov_kernel : CovKernel,
            noise=np.float32(1e-7)
        ):
        self.mean_kernel = mean_kernel
        self.cov_kernel = cov_kernel
        self.noise = noise

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        print(f"Predicting for {X} points")

        print("[Computing K]: ", end="")
        K = self.cov_kernel(self.X_train, self.X_train) + self.noise * np.eye(len(self.X_train))
        print("[Computing K_s]: ", end="")
        K_s = self.cov_kernel(self.X_train, X)
        print("[Computing K_ss]: ", end="")
        K_ss = self.cov_kernel(X, X) + self.noise

        print("K shape: ", K.shape)
        print("K_s shape: ", K_s.shape)
        print("K_ss shape: ", K_ss.shape)

        mu = K_s.T @ np.linalg.inv(K) @ (self.y_train - self.mean_kernel(self.X_train)) + self.mean_kernel(np.array([1]))
        sig = K_ss - K_s.T @ np.linalg.inv(K) @ K_s

        return mu, sig

