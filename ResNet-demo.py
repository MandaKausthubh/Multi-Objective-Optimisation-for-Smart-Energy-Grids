
import warnings
warnings.filterwarnings("ignore")

import os
import sys
sys.path.append('../')

import time
import numpy as np 
import pandas as pd

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
from torch.quasirandom import SobolEngine

# BoTorch / GPyTorch / SAASBO
from botorch.models import SaasFullyBayesianSingleTaskGP
from botorch.fit import fit_fully_bayesian_model_nuts
from botorch.acquisition import qUpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples

# Custom ResNet model
from ResNet_model import *  # Assumes NN_prediction is defined here
from torch.utils.data import DataLoader, TensorDataset

# Load your data (make sure this is defined)
# Example placeholder

data = pd.read_csv('data/Berlin_wind_multiclass.csv')
print(data.head())
print(data.shape)
# Create a DataFrame that includes all columns except 'X50Hertz..MW.' / 'TransnetBW..MW.'
target_column = 'X50Hertz..MW.'    #  'X50Hertz..MW.' or 'TransnetBW..MW.'
features = data.drop(columns=[target_column])

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler to the feature subset and transform it
scaled_features = scaler.fit_transform(features)

# Convert the scaled features back to a DataFrame with column names
scaled_features_df = pd.DataFrame(scaled_features, columns=features.columns)

# Combine the scaled features with the "X50Hertz..MW" column
scaled_data = pd.concat([scaled_features_df, data[target_column]], axis=1)

# Define the number of classes to divide
num_classes = 5

# Case 1: Divide the target column into 5 classes based on quantiles
# Create one-hot encoded columns for class labels
class_labels = pd.qcut(data[target_column], q=num_classes, labels=False)
class_columns = pd.get_dummies(class_labels, prefix='class')

# Concatenate the one-hot encoded columns with the original DataFrame
data_with_classes = pd.concat([data, class_columns], axis=1)
y = data_with_classes[['class_0', 'class_1', 'class_2', 'class_3', 'class_4']]  

# Make a list of features to drop
dropcols = [target_column, 'class_0', 'class_1', 'class_2', 'class_3', 'class_4']  
X = data_with_classes.drop(columns = dropcols, axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)  
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state = 42)  


train_features = torch.tensor(X_train.values, dtype=dtype, device=device)
train_targets = torch.tensor(y_train.values, dtype=dtype, device=device)
val_features = torch.tensor(X_val.values, dtype=dtype, device=device)
val_targets = torch.tensor(y_val.values, dtype=dtype, device=device)
test_features = torch.tensor(X_test.values, dtype=dtype, device=device)
test_targets = torch.tensor(y_test.values, dtype=dtype, device=device)

train_ds = TensorDataset(train_features, train_targets)
val_ds = TensorDataset(val_features, val_targets)
test_ds = TensorDataset(test_features, test_targets)

bs = 1000 
train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=True)
test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=True)


training_data = {
    'features' : X_train,
    'labels' : y_train
}

testing_data = {
    'features' : X_test,
    'labels' : y_test
}

validation_data = {
    'features' : X_val,
    'labels' : y_val
}


fairness_feature = 'GHI'    # fairness_feature is only used in multi-class classification
training_data = (training_data['features'], training_data['labels'])
testing_data = (testing_data['features'], testing_data['labels'])
validation_data = (validation_data['features'], validation_data['labels'])

def black_box_function(x, weights):
    try:
        x_np = x.detach().cpu().numpy().reshape(-1)
        y_pred, _, model = NN_prediction(x_np, test_features, train_loader, val_loader)
        sparsity = compute_model_density(model)
        acc = sklearn.metrics.accuracy_score(y_test, y_pred)
        objectives = torch.tensor([1.0 - acc, sparsity])  # minimize (1 - accuracy)
        return torch.Tensor([acc, sparsity, (objectives @ weights).item()])  # Weighted sum of objectives
    except:
        print("Unexpected eval error", sys.exc_info()[0])
        return torch.tensor([1.0])  # Worst case if evaluation fails

# SAASBO settings
dim = 20  # Modify if your hyperparameter space is different
bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])

n_init = 16

weights = torch.tensor([
    [0.0, 1.0],
    [0.5, 0.5],
    [0.2, 0.8],
    [0.8, 0.2],
    [0.3, 0.7],
    [0.7, 0.3],
    [0.4, 0.6],
    [0.6, 0.4],
    [0.9, 0.1],
    [1.0, 0.0],
])  # Weights for objectives
train_x = draw_sobol_samples(bounds=bounds, n=1, q=n_init).squeeze(0)
train_y = torch.vstack([black_box_function(x, weights[0]).unsqueeze(0) for x in train_x])
# train_y = train_y[:, 2].unsqueeze(1)  # Only keep the objective value
print(train_x.shape, train_y.shape)

train_y_scalar = train_y[:, 2].unsqueeze(1)
model = SaasFullyBayesianSingleTaskGP(train_x, train_y_scalar)

n_iter = 10
sparse = torch.zeros(n_iter)
accuracy = torch.zeros(n_iter)

n_weights = weights.shape[0]
n_iter = 10

all_sparse = []
all_accuracy = []

for j in range(n_weights):
    print(f"\n=== Optimizing for weight vector {weights[j]} ===")

    # Fresh initialization for each run
    train_x = draw_sobol_samples(bounds=bounds, n=1, q=n_init).squeeze(0)
    train_y = torch.vstack([black_box_function(x, weights[j])[-1].unsqueeze(0) for x in train_x])

    # Fit initial model
    model = SaasFullyBayesianSingleTaskGP(train_x, train_y)
    fit_fully_bayesian_model_nuts(model)

    for i in range(n_iter):
        model.eval()
        acq_func = qUpperConfidenceBound(model=model, beta=0.1)

        new_x, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=512,
        )

        acc, spar, new_y = black_box_function(new_x.squeeze(0), weights[j])
        new_y = new_y.unsqueeze(0)  # Shape [1]
        
        train_x = torch.cat([train_x, new_x], dim=0)
        train_y = torch.cat([train_y, new_y], dim=0)

        model = SaasFullyBayesianSingleTaskGP(train_x, train_y)
        fit_fully_bayesian_model_nuts(model)

    # Evaluate best result for this weight
    best_idx = train_y.argmin()
    best_input = train_x[best_idx]

    # Re-evaluate full objectives at best input
    acc, spar, _ = black_box_function(best_input, weights[j])
    all_sparse.append(spar)
    all_accuracy.append(acc)

    print(f"Best input for weight {weights[j]}:")
    print(f"  Accuracy = {acc:.4f}")
    print(f"  Sparsity = {spar:.4f}")


#Plotting results:
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
# Plot accuracy vs sparsity
plt.plot(sparse.numpy(), accuracy.numpy(), marker='o')
plt.xlabel('Sparsity')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Sparsity')
plt.grid()
plt.show()
#Saving the plot
plt.savefig('accuracy_vs_sparsity.png')

best_idx = train_y.argmin()
print("Best accuracy:", 1.0 - train_y[best_idx].item())
print("Best input:", train_x[best_idx])
