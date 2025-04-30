from abc import abstractmethod
import numpy as np
from scipy.stats import norm



# -------- Covariance Kernels --------
class CovKernel():
    def __init__(self, length_scale=1.0):
        self.length_scale = length_scale

    @abstractmethod
    def __call__(self, x1, x2) -> np.ndarray:
        pass

class RBFKernel(CovKernel):
    def __init__(self, ratios=None, length_scale=1.0):
        super().__init__(length_scale)
        if ratios is None:
            self.ratios = np.ones(1)
        else:
            self.ratios = ratios

    def Kernel2Sets(self, x1, x2):
        answer = np.zeros((x1.shape[0], x2.shape[0]))
        for i in range(x1.shape[0]):
            for j in range(x2.shape[0]):
                answer[i, j] = np.exp(np.linalg.norm(x1[i] - x2[j]) ** 2 / (2 * self.length_scale ** 2))
        return answer

    def KernelSetVector(self, x1, v1):
        return np.exp(np.linalg.norm(x1 - v1, axis=1) ** 2 / (2 * self.length_scale ** 2))

    def KernelVec2Vec(self, v1, v2):
        return np.exp(np.linalg.norm(v1 - v2) ** 2 / (2 * self.length_scale ** 2))

    def __call__(self, x1, x2) -> np.ndarray:
        if x1.ndim == 1 and x2.ndim == 1:
            return self.KernelVec2Vec(x1, x2)
        elif x1.ndim == 2 and x2.ndim == 2:
            return self.Kernel2Sets(x1, x2)
        elif x1.ndim == 2 and x2.ndim == 1:
            return self.KernelSetVector(x1, x2)
        elif x1.ndim == 1 and x2.ndim == 2:
            return self.KernelSetVector(x2, x1)
        else:
            raise ValueError("Invalid input dimensions for kernel computation.")




# -------- Mean Kernels --------
class MeanKernel():
    def __init__(self, mean=0.0):
        self.mean = mean

    def __call__(self, x):
        return np.full(x.shape[0], self.mean)


# -------- Gaussian Process --------
class GaussianProcess():
    def __init__(self,
                 mean_kernel : MeanKernel,
                 cov_kernel : CovKernel,
                 noise=np.float32(1e-10)
                 ):
        self.mean_kernel = mean_kernel
        self.cov_kernel = cov_kernel
        self.noise = noise

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        K = self.cov_kernel(self.X_train, self.X_train) + self.noise * np.eye(len(self.X_train))
        K_s = self.cov_kernel(self.X_train, X)
        K_ss = self.cov_kernel(X, X) + 1e-8 * np.eye(len(X))

        mu = K_s.T @ np.linalg.inv(K) @ (self.y_train - self.mean_kernel(self.X_train)) + self.mean_kernel(X)
        cov = K_ss - K_s.T @ np.linalg.inv(K) @ K_s

        return mu, cov

    def sample(self, X, n_samples=1):
        mu, cov = self.predict(X)
        samples = np.random.multivariate_normal(mu, cov, n_samples)
        return samples




# -------- Acquisition Functions --------
class AcquisitionFunction():
    def __init__(self, gp: GaussianProcess):
        self.gp = gp

    @abstractmethod
    def __call__(self, X):
        pass

class ExpectedImprovement(AcquisitionFunction):

    def __init__(self, gp: GaussianProcess, best_y: float):
        super().__init__(gp)
        self.best_y = best_y

    def __call__(self, X):
        mu, cov = self.gp.predict(X)
        sigma = np.sqrt(np.diag(cov))

        with np.errstate(divide='warn'):
            Z = (mu - self.best_y) / sigma
            ei = (mu - self.best_y) * norm.cdf(Z) + sigma * norm.pdf(Z)

        return ei


# --------- Bayesian Optimizer ---------
class BayesianOptimizer():
    def __init__(self,
                 gp: GaussianProcess,
                 acquisition_function: AcquisitionFunction,
                 bounds: np.ndarray,
                 objective_function,
                 n_initial_points: int = 5,
                 ):
        self.gp = gp
        self.acquisition_function = acquisition_function
        self.bounds = bounds
        self.n_initial_points = n_initial_points
        self.objective_function = objective_function

    def _find_next_point(self):
        # Generate random points within the bounds
        X_random = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (1000, self.bounds.shape[0]))

        # Evaluate the acquisition function at these points
        acquisition_values = self.acquisition_function(X_random)

        # Select the point with the highest acquisition value
        best_index = np.argmax(acquisition_values)
        return X_random[best_index]

    def optimize(self, n_iterations: int):
        # Initialize with random points
        X_init = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.n_initial_points, self.bounds.shape[0]))
        y_init = np.array([self.objective_function(x) for x in X_init])
        self.gp.fit(X_init, y_init)

        for _ in range(n_iterations):
            # Find the next point to sample
            X_next = self._find_next_point()
            y_next = self.objective_function(X_next)

            # Update the GP with the new point
            self.gp.fit(np.vstack((self.gp.X_train, X_next)), np.hstack((self.gp.y_train, y_next)))

        return X_next, y_next
