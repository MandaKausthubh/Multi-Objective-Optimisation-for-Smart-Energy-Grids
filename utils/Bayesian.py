import numpy as np
from utils.GaussianProcesses import GaussianProcess
from utils.Acquisition import AcquisitionFunction

# -------- Bayesian Optimization --------
class BayesianOptimization():
    def __init__(
            self,
            bounds,
            acquisition_function: AcquisitionFunction,
            gp: GaussianProcess,
            x_init = [],
            y_init = [],
            n_iter=25
        ):
        self.objective_function = acquisition_function.objective_function
        self.bounds = bounds
        self.acquisition_function = acquisition_function
        self.gp = gp
        self.n_iter = n_iter
        self.X_init = x_init
        self.y_init = y_init
        print("Inserting initial points")
        self.acquisition_function.InsertSetOfNewPoints(x_init, y_init)


    def optimize(self):
        # Fit the GP model with the initial points
        self.gp.fit(self.X_init, self.y_init)

        print("[Starting optimization]")

        # Iterate to find the optimal point
        for _ in range(self.n_iter):
            x_next, y_next = self.acquisition_function(method="L-BFGS-B")
            print(f"Next point: {x_next}, Function value: {y_next}")
            print(f"Points so far: {self.X_init}, Function values so far: {self.y_init}")
            # Update the GP model with the new point
            self.acquisition_function.InsertNewPoints(x_next, y_next)
            self.X_init = np.vstack((self.X_init, x_next))
            self.y_init = np.append(self.y_init, y_next)
            self.gp.fit(self.X_init, self.y_init)
        print("[Optimization finished]")

        return self.X_init, self.y_init
