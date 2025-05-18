import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import sklearn
from torchmetrics import MeanSquaredError, MeanSquaredLogError
from sklearn.metrics import mean_squared_error

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.double



# ------ NN metrics (Pytorch) ------
def mean_squared_error_NOMAD(y_pred, y_true): 
    mse_metric = MeanSquaredError().to(dtype=dtype, device=device)
    mse_score = mse_metric(y_pred, y_true).unsqueeze(-1)
    mse_score = mse_score.detach().cpu().numpy()

    # Convert NumPy array to numpy.float64
    mse_score_np = np.float64(mse_score)
    #print('mse_score_np', mse_score_np.shape)
    return mse_score_np


def mean_bias_error_NOMAD(y_pred, y_true):
    n = y_true.size()[0]  # Number of samples
    mbe_metric = (torch.abs(torch.sum(y_true - y_pred)) / n).unsqueeze(-1)
    mbe_score = mbe_metric.detach().cpu().numpy()

    # Convert NumPy array to numpy.float64
    mbe_score_np = np.float64(mbe_score)   
    #print('mbe', mbe_score_np)
    return mbe_score_np


def accuracy_NOMAD(y_pred, y_true):
    correct_predictions = (y_pred == y_true).all(dim=1).sum().item()
    total_samples = y_true.size(0)
    accuracy = correct_predictions / total_samples
    accuracy_error = np.round(1 - accuracy, decimals=4).astype(float)
    return accuracy_error


def dpd_score_NOMAD(y_pred, y_true, group=None):
    if group is None or group.size == 0:
        group = X_test[fairness_feature].values
    
    # Convert PyTorch tensors to NumPy arrays
    y_pred = y_pred.detach().cpu().numpy()
    
    # Use list comprehension to map elements less than 0.5 to 0, and greater than or equal to 0.5 to 1
    group_new_arr = np.array([0 if i < 0.5 else 1 for i in group])
    
    # Compute the predicted outcome for the protected group
    p_protected = np.sum(y_pred[group_new_arr == 1][:,1:])  

    # Compute the predicted outcome for the non-protected group
    p_non_protected = np.sum(y_pred[group_new_arr == 0][:, 0])   # [:, 0]
    
    # Sample points that correspond to 1’s and 0’s
    count_one = np.count_nonzero(group_new_arr)
    count_zero = len(group_new_arr) - np.count_nonzero(group_new_arr)

    # Compute the DPD score
    dpd = np.round(np.abs((p_protected/count_one) - (p_non_protected/count_zero)), decimals=4).astype(float) 
    return dpd

def sparsity_NOMAD(model, threshold=1e-1):
    # Calculate sparsity of the model
    total_parameters = 0
    zero_near_zero_weights = 0

    for name, param in model.named_parameters():
        if 'weight' in name:  # Only consider weight matrices
            total_parameters += param.numel()
            zero_near_zero_weights += torch.sum(torch.abs(param) <= threshold).item()

    sparsity = zero_near_zero_weights / total_parameters
    density = np.round(1- sparsity, decimals=4).astype(float)   # density = 1 - sparsity
    return density



# ------ XGBoost metrics ------

def mean_squared_error_xgb(y_pred, y_true): 
    mse_score = mean_squared_error(y_true, y_pred)
    mse_score = np.float64(mse_score)
    return mse_score


def mean_bias_error_xgb(y_pred, y_true):
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    n = y_pred.shape[0]  # Number of samples
    mbe_metric = np.abs(np.sum(y_true - y_pred)) / n
    mbe_metric = np.float64(mbe_metric)    
    return mbe_metric


def accuracy_xgb(y_pred, y_true):
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    accuracy_error = np.round(1 - accuracy, decimals=4).astype(float)
    return accuracy_error

def dpd_score_xgb(y_pred, y_true, group=None):  # y_pred is a numpy array of prediction labels       
    if group is None or group.size == 0:
        group = X_test[fairness_feature].values
    
    y_pred = y_pred.astype(int)
    n_classes = (np.max(y_pred) + 1).astype(int)
    one_hot_labels = np.eye(n_classes)[y_pred]

    # Use list comprehension to map elements less than 0.5 to 0, and greater than or equal to 0.5 to 1
    group_new_arr = np.array([0 if i < 0.5 else 1 for i in group])

    # Compute the predicted outcome for the protected group
    p_protected = np.sum(one_hot_labels[group_new_arr == 1][:, 1:])   

    # Compute the predicted outcome for the non-protected group
    p_non_protected = np.sum(one_hot_labels[group_new_arr == 0][:, 0])   
    
    # Sample points that correspond to 1’s and 0’s
    count_one = np.count_nonzero(group_new_arr)
    count_zero = len(group_new_arr) - np.count_nonzero(group_new_arr)

    # Compute the DPD score
    dpd = np.round(np.abs((p_protected/count_one) - (p_non_protected/count_zero)), decimals=4).astype(float) 
    return dpd

def average_depth_xgb(model):
    # Get the dump of all trees
    trees = model.get_dump()

    total_depth = 0
    for tree in trees:
        # Count depth of each tree
        depth = 0
        for line in tree.split('\n'):
            if line.startswith('\t'):
                # Count the number of tabs, which indicate depth
                current_depth = line.count('\t')
                depth = max(depth, current_depth)

        total_depth += depth

    # Calculate the average depth
    average_depth = total_depth / len(trees) if trees else 0
    average_depth = np.round(average_depth, decimals=4).astype(float)
    return average_depth


# ------ KNN metrics ------
# Here is how to implement SDE
# Please note that most of the code is referenced from the code provided by the paper "Measuring Bias and Fairness in Multiclass Classification".

def class_level_stats(y_true, y_pred):
  conf_matrix = confusion_matrix(y_true, y_pred)

  FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
  FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
  TP = np.diag(conf_matrix)
  TN = conf_matrix.sum() - (FP + FN + TP)

  FPR = FP / (FP  + TN)
  FNR = FN / (FN + TP)
  return FPR, FNR

def my_special_variance(error_rates):
  assert error_rates.ndim == 2, 'Error rates must be 2 dimensional'
  my_special_mean = error_rates.sum(axis=0)/[len(error_rates), len(error_rates)]
  sum_of_squares = 0
  for e in error_rates:
    sum_of_squares += ((my_special_mean[0]-e[0])**2 + (my_special_mean[1]-e[1])**2)
  return sum_of_squares/len(error_rates)


def dis_from_sym(p):
    return np.mean(np.abs(p[:, 0] - p[:, 1]))

def get_CEV_SDE(y_pred, y_true, group=None):
    all_test_class_labels = np.argmax(y_true, axis=1)
    all_pred_class_labels = np.argmax(y_pred, axis=1)
    
    all_FPR, all_FNR = class_level_stats(all_test_class_labels, all_pred_class_labels)
    
    if group is None or group.size == 0:
        group = X_test[fairness_feature].values

    # Use list comprehension to map elements less than 0.5 to 0, and greater than or equal to 0.5 to 1
    group_new_arr = np.array([0 if i < 0.5 else 1 for i in group])
    TP_indices = np.where(group_new_arr == 1)[0]
    test_subset = y_true.iloc[TP_indices]
    pred_subset = y_pred[TP_indices]
    
    # The class label will be the index of '1' in each row
    fair_test_class_labels = np.argmax(test_subset, axis=1)
    fair_pred_class_labels = np.argmax(pred_subset, axis=1)

    fair_FPR, fair_FNR = class_level_stats(fair_test_class_labels, fair_pred_class_labels)
    
    FPR_change = (fair_FPR - all_FPR)/all_FPR 
    FNR_change = (fair_FNR - all_FNR)/all_FNR 

    # make class points
    points = np.dstack((FPR_change, FNR_change))[0]

    CEV = my_special_variance(points)
    SDE = dis_from_sym(points)
    return CEV, SDE