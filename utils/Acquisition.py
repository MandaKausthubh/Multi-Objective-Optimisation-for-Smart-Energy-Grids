from typing import Tuple
import numpy as np
from abc import abstractmethod
from scipy.stats import norm
from scipy.optimize import minimize
from utils.GaussianProcesses import GaussianProcess


# -------- Acquisition Functions --------
class AcquisitionFunction():
    def __init__(self, gp: GaussianProcess, objective_function=None):
        self.gp = gp
        self.objective_function = objective_function

    @abstractmethod
    def __call__(self, method) -> Tuple[np.float64, np.float64]:
        """
        Call the acquisition function to find the next point to sample, and the corresponding function value.
        """
        pass

    @abstractmethod
    def InsertNewPoints(self, X, y) -> None:
        """
        Insert new points into the acquisition function.
        This method is optional and can be overridden by subclasses.
        """
        pass

    @abstractmethod
    def InsertSetOfNewPoints(self, X, y) -> None:
        """
        Insert a set of new points into the acquisition function.
        This method is optional and can be overridden by subclasses.
        """
        pass



class ExpectedImprovement(AcquisitionFunction):

    def __init__(self, gp: GaussianProcess, bounds, objective_function, X = [], y = [], min_max = "min"):
        super().__init__(gp, objective_function)
        self.X_init, self.y_init = X, y
        self.objective_function = objective_function
        self.bounds = bounds

        assert min_max in ["min", "max"]
        if len(X) == 0 or len(y) == 0:
            self.y_best = np.inf if min_max == "min" else -np.inf
        else:
            self.y_best = min(y) if min_max == "min" else max(y)

        self.mode = min_max
        self.gp.fit(X, y)

    def InsertNewPoints(self, X, y):
        self.X_init.append(X); self.y_init.append(y)
        self.y_best = min(y, self.y_init[-1]) if self.mode == "min" else max(self.y_init[-1], y)
        self.gp.fit(self.X_init, self.y_init)

    def InsertSetOfNewPoints(self, X, y):
        self.X_init.extend(X); self.y_init.extend(y)
        self.gp.fit(self.X_init, self.y_init)
        if self.mode == "min":
            self.y_best = min(self.y_init)
        else:
            self.y_best = max(self.y_init)

    def Imporvement(self, x, y = None):
        val = self.objective_function(x) if y is None else y
        if self.mode == "min": 
            return max(0, self.y_best - val)
        else:
            return max(0, val - self.y_best)

    def ExpectedImporvement(self, x):
        print(f"ExpectedImporvement: {x}")
        mu, sigma = self.gp.predict(x)
        # sigma = np.sqrt(np.diag(cov))
        print(f"mu: {mu}, sigma: {sigma}")
        with np.errstate(divide='warn', invalid='ignore'):
            Z = (mu - self.y_best) / (sigma + 1e-9)
            ei = (mu - self.y_best) * norm.cdf(Z) + sigma * norm.pdf(Z)
        return ei

    def __call__(self, method="L-BFGS-B") -> Tuple[np.float64, np.float64]:
        x0 = [np.random.uniform(lower, upper) for (lower, upper) in self.bounds]
        res = 0
        if self.mode == "min":
            res = minimize(lambda x: -self.ExpectedImporvement(x), x0, bounds=self.bounds, method=method)
            return res.x, self.objective_function(res.x)
        else:
            res = minimize(lambda x: self.ExpectedImporvement(x), x0, bounds=self.bounds, method=method)
            return res.x, self.objective_function(res.x)

