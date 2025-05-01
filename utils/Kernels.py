from abc import abstractmethod
import numpy as np

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
        x1 = np.asarray(x1)
        x2 = np.asarray(x2)
        answer = np.zeros((x1.shape[0], x2.shape[0]))
        for i in range(x1.shape[0]):
            for j in range(x2.shape[0]):
                answer[i, j] = np.exp(np.linalg.norm(x1[i] - x2[j]) ** 2 / (2 * self.length_scale ** 2))
        return answer

    def KernelSetVector(self, x1, v1):
        x1 = np.asarray(x1)
        return np.exp(np.linalg.norm(x1 - v1, axis=1) ** 2 / (2 * self.length_scale ** 2))

    def KernelVec2Vec(self, v1, v2):
        v1 = np.asarray(v1)
        return np.exp(np.linalg.norm(v1 - v2) ** 2 / (2 * self.length_scale ** 2))

    def __call__(self, x1, x2) -> np.ndarray:
        x1 = np.asarray(x1)
        x2 = np.asarray(x2)
        print(f"Input dimensions: x1: {x1.ndim}, x2: {x2.ndim}")
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
        x = np.asarray(x)
        return np.full(x.shape[0], self.mean)










