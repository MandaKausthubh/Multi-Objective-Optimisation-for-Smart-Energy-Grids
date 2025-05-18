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
        """
        Predict the mean and covariance of the Gaussian process at the given points.
        """
        X = np.asarray(X)  # Query point

        K = self.cov_kernel(self.X_train, self.X_train) + self.noise * np.eye(len(self.X_train))
        print(f"Covariance matrix K: {K.shape}")

        K_s = self.cov_kernel(self.X_train, X)
        print(f"Covariance matrix K_s: {K_s.shape}")

        K_ss = self.cov_kernel(X, X) + self.noise #+ 1e-8 * np.eye(len(X))
        print(f"Covariance matrix K_ss: {K_ss.shape}")

        inv_K = np.linalg.inv(K)

        mu_f_query =  K_s.T @ inv_K @ (self.y_train - self.mean_kernel(self.X_train)) + self.mean_kernel(np.array([1]))
        print(f"Mean prediction mu_f_query: {mu_f_query.shape}")

        cov_f_query = K_ss - K_s.T @ inv_K @ K_s
        print(f"Covariance prediction cov_f_query: {cov_f_query.shape}")

        return mu_f_query, cov_f_query


