# The focus of our research is multi-objective optimization in machine learning, 
# i.e., we aim to minimize multiple objectives simultaneously. 
# This demo shows the performance of weighted BO, weighted MADS, and joint method on a ResNet model. 
# For the regression problem, the objectives are MSE, MBE, and density. 
# For multi-class classification, the objectives are accuracy error, density, and DPD. 
# Note that this demo uses CUDA by default, and it takes about 300 minutes to execute this demo at one time.

import warnings
warnings.filterwarnings("ignore")
# System Level Items -- paths etc. 
import os
import sys
sys.path.append('../')
from io import StringIO
import time
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import PyNomad
import smt
from smt.surrogate_models import KRG
from smt.applications.ego import EGO
from smt.sampling_methods import LHS
from smt.utils.design_space import (
    DesignSpace,
    FloatVariable,
    IntegerVariable,
)
from pygmo import hypervolume

from Metrics import *
from ResNet_model import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.double
print(device)


torch.cuda.empty_cache()

# Get arguments -- dataset used (solar in our case)
data_set = sys.argv[0]
task_type = 'multiclass'   # 'regression' or ''binary' or 'multiclass' or 'multilabel'


# ------ Solar regression dataset ------
#data = pd.read_csv('data/Stuttgart_solar_regression.csv')
#target_column = 'TransnetBW..MW.'  # 'X50Hertz..MW.' / 'TransnetBW..MW.'

# Regression
#y = data[[target_column]]   # 'power'  -- Berlin
#dropcols = [target_column]   # 'power'  -- Berlin
#X = data.drop(columns = dropcols, axis=0)


# ------ Wind multi-class classification dataset ------
data = pd.read_csv('data/Berlin_wind_multiclass.csv')

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


# ------
# data
# ------
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

# Hyperparameters for the model
dim = 3

min_number_res_block = 1
max_number_res_block = 15

min_neurons_per_layer = 8
max_neurons_per_layer = 128

min_dropout_rate = 0.01
max_dropout_rate = 0.3


# Parameter setting for weighted BO
# xlimits: float (continuous) variables: array-like of size nx x 2 (i.e. lower, upper bounds)
xlimits = np.array([[min_number_res_block, max_number_res_block], 
                    [min_neurons_per_layer, max_neurons_per_layer],
                    [min_dropout_rate, max_dropout_rate]])

criterion='EI' #'EI' or 'SBO' or 'LCB'

#number of points in the initial x_init
ndoe = 10   # (at least ndim+1)

#design_space = DesignSpace(xlimits)
design_space = DesignSpace ([
    IntegerVariable (min_number_res_block, max_number_res_block),
    IntegerVariable (min_neurons_per_layer, max_neurons_per_layer), 
    FloatVariable (min_dropout_rate, max_dropout_rate)])

sampling = LHS(xlimits=xlimits, random_state=42)

#EGO call
sm = KRG(design_space=design_space, n_start=25, print_global=False)


# Parameter setting for weighted MADS
# lower boundary & upper boundary
lb = [min_number_res_block, min_neurons_per_layer, min_dropout_rate]
ub = [max_number_res_block, max_neurons_per_layer, max_dropout_rate]

# starting point x0 is obtained by BO


reference_point = np.array([1+1e-7, 1+1e-7, 1+1e-7])
hvs_joint_method_SOO, computational_time_joint_method_SOO = [], []

    
# Define the weights for the weighted sum of objectives
sets_of_weights = [
    [0, 1/3, 2/3],
    [0, 2/3, 1/3],
    [1/3, 2/3, 0],
    [2/3, 1/3, 0],
    [1/3, 0, 2/3],
    [2/3, 0, 1/3],
    [1/3, 1/3, 1/3],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
]

def find_index(weight_set):
    for i, known_set in enumerate(sets_of_weights):
        if weight_set == known_set:
            return int(i)
    return "Unknown set of weights"


# ------ weighted BO ------
def single_obj_optimization_only_BO(x_init):
    dpd_score = []
    density = []
    mse = []
    mbe = []
    accuracy_error = []
    
    for i in range(x_init.shape[0]):
        y_pred, training_time_list, ResNet_trained_model = NN_prediction(x_init[i], test_features, train_loader, val_loader, task_type=task_type)
        
        # Objectives for regression
        #new_mse = mean_squared_error_NOMAD(y_pred, test_targets)
        #mse.append(new_mse)
        
        #new_mbe = mean_bias_error_NOMAD(y_pred, test_targets)
        #mbe.append(new_mbe)
        
        #new_sparsity = sparsity_NOMAD(ResNet_trained_model)
        #density.append(new_sparsity)
        
        # Objectives for multi-class classification
        new_accuracy = accuracy_NOMAD(y_pred, test_targets)
        accuracy_error.append(new_accuracy)

        new_sparsity = sparsity_NOMAD(ResNet_trained_model)
        density.append(new_sparsity)
        
        new_dpd = dpd_score_NOMAD(y_pred, test_targets, group=X_test[fairness_feature].values)
        dpd_score.append(new_dpd)
    
    # Objectives for regression
    #mse_arr = np.round(np.array(mse), decimals=4).astype(np.float64)
    #mbe_arr = np.round(np.array(mbe), decimals=4).astype(np.float64)
    #density_arr = np.round(np.array(density), decimals=4).astype(np.float64)
    
    # Objectives for multi-class classification
    accerror_arr = np.round(np.array(accuracy_error), decimals=4).astype(np.float64)  
    density_arr = np.round(np.array(density), decimals=4).astype(np.float64)    
    dpd_arr = np.round(np.array(dpd_score), decimals=4).astype(np.float64)

    sum_objectives_array = (weight_1 * accerror_arr + weight_2 * density_arr + weight_3 * dpd_arr).reshape(-1, 1)
    combined_new_array = np.column_stack((accerror_arr, density_arr, dpd_arr))
    
    if x_init.shape[0] == ndoe or x_init.shape[0] == 1:
        # write the HP configs into file
        paths='.'
        Weighted_BO_SOO_outputs_filename = f"weighted_BO_SOO_outputs_{weight_idx}.txt"
        Weighted_BO_SOO_outputs_file = open(os.path.join(paths, Weighted_BO_SOO_outputs_filename), "a")
        Weighted_BO_SOO_outputs_file.write(str(combined_new_array)+'\n' )
        Weighted_BO_SOO_outputs_file.close()
        
        # write the objective values into file   
        Weighted_BO_HP_filename = f"weighted_BO_HP_{weight_idx}.txt"
        HP_candidates_file = open(os.path.join(paths, Weighted_BO_HP_filename), "a")
        HP_candidates_file.write(str(x_init)+'\n' )
        HP_candidates_file.close()       
    return sum_objectives_array

# ------ Perform weighted BO ------
def run_weighted_BO(x_init, max_iterations=1000, time_budget=600):
    iteration = 0
    x_init_list = []
    next_record_time = 60  # Set the first recording time to 30 seconds
    
    start_time_overall = time.time()
    while iteration < max_iterations:  
        # EGO call
        ego = EGO(
        n_iter = n_iter,
        criterion = criterion,
        xdoe = x_init,
        surrogate = sm,
        n_start = 30)  #to do multistart for maximizing the acquisition function
        
        x_opt, y_opt, ind_best, x_data, y_data = ego.optimize(fun = single_obj_optimization_only_BO)
        x_init = np.vstack((x_opt, x_data[-3:,:]))
        x_init_list.extend(x_opt)

        BO_computational_time_overall = time.time() - start_time_overall

        # Check if it's time to record the result
        if BO_computational_time_overall >= next_record_time:
            Weighted_BO_outputs_filename = f"weighted_BO_SOO_outputs_{weight_idx}.txt"
            with open(Weighted_BO_outputs_filename, "r") as file:
                lines = file.read().split()
            cleaned_content = [float(item.strip("[],")) for item in lines if item.strip("[],")]
            weighted_BO_outputs_arr = np.array(cleaned_content).reshape(-1, 3)

            weighted_BO_HP_filename = f"weighted_BO_HP_{weight_idx}.txt"
            with open(weighted_BO_HP_filename, "r") as file:
                lines = file.read().split()
            cleaned_content = [float(item.strip("[],")) for item in lines if item.strip("[],")]
            weighted_BO_HP_arr = np.array(cleaned_content).reshape(-1, 3)
            
            weight_idx_arr = np.full((weighted_BO_outputs_arr.shape[0], 1), weight_idx)
            BO_results_arr = np.column_stack((weight_idx_arr, weighted_BO_outputs_arr, weighted_BO_HP_arr))
            
            # Save the array to a text file
            paths='./BO_results/ResNet/'
            Weighted_BO_merge_file_filename = f"{weight_idx}_weighted_BO_FuncVal_HP_{next_record_time}.txt"
            Weighted_BO_merge_file = os.path.join(paths, Weighted_BO_merge_file_filename)
            np.savetxt(Weighted_BO_merge_file, BO_results_arr, fmt='%.4f', delimiter=' ')

            next_record_time += 60  # Set the next recording time
            
        # Check if the time budget has been exceeded
        if BO_computational_time_overall > time_budget:
            print("Time budget exceeded. Stopping execution.")
            break
        
        iteration += 1
    return x_init_list
        
        
# ------ weighted MADS ------
def bb_only_MADS(x):
    try:
        dim = x.size()
        x_init = [x.get_coord(i)  for i in range(dim)]
        
        y_pred, training_time_list, ResNet_trained_model = NN_prediction(x_init, test_features, train_loader, val_loader, task_type=task_type)
        
        # Objectives for regression
        #mse = mean_squared_error_NOMAD(y_pred, test_targets)   
        #mbe = mean_bias_error_NOMAD(y_pred, test_targets) 
        #density = sparsity_NOMAD(new_net)
        
        # Objectives for multi-class classification
        accuracy_error = accuracy_NOMAD(y_pred, test_targets) 
        density = sparsity_NOMAD(ResNet_trained_model)
        dpd = dpd_score_NOMAD(y_pred, test_targets, group=X_test[fairness_feature].values)
        
        sum_objectives = weight_1 * accuracy_error + weight_2 * density + weight_3 * dpd
        combined_new_array = np.column_stack((accuracy_error, density, dpd))

        x_init_arr = np.array(x_init).reshape(1, -1)
        
        # write the HP configs into file
        paths='.'
        Weighted_MADS_HP_candidates_file_filename = f"weighted_MADS_HP_candidates_{weight_idx}.txt"
        HP_candidates_file = open(os.path.join(paths, Weighted_MADS_HP_candidates_file_filename), "a")
        HP_candidates_file.write(str(x_init_arr)+'\n' )
        HP_candidates_file.close()
        
        # write the HP configs into file
        paths='.'
        Weighted_MADS_SOO_outputs_filename = f"weighted_MADS_SOO_outputs_{weight_idx}.txt"
        Weighted_MADS_SOO_outputs_file = open(os.path.join(paths, Weighted_MADS_SOO_outputs_filename), "a")
        Weighted_MADS_SOO_outputs_file.write(str(combined_new_array)+'\n' )
        Weighted_MADS_SOO_outputs_file.close()

        x.setBBO(str(sum_objectives).encode("UTF-8"))
    except:
        print("Unexpected eval error", sys.exc_info()[0])
        return 0
    return 1 # 1: success 0: failed evaluation

# ------ Perform weighted MADS ------
def run_weighted_MADS(x_init, max_iterations=1000, time_budget=600):
    iteration = 0
    x_init_list = []  
    next_record_time = 60  # Set the first recording time to 60 seconds
    
    start_time_overall = time.time()
    while iteration < max_iterations:
        print('iteration', iteration) 
        params_only_MADS = ["DIMENSION 3", "BB_OUTPUT_TYPE OBJ", "MAX_BB_EVAL 6", "DISPLAY_DEGREE 2", "DISPLAY_ALL_EVAL false", "DISPLAY_STATS BBE OBJ",
        "DIRECTION_TYPE N+1 UNI"]
        
        result = PyNomad.optimize(bb_only_MADS, x_init, lb, ub, params_only_MADS)
        x_init = result['x_best']
        x_init_list.extend(x_init)
        
        MADS_computational_time_overall = time.time() - start_time_overall
        
        # Check if it's time to record the result
        if MADS_computational_time_overall >= next_record_time:
            Weighted_MADS_SOO_outputs_filename = f"weighted_MADS_SOO_outputs_{weight_idx}.txt"
            with open(Weighted_MADS_SOO_outputs_filename, "r") as file:
                lines = file.read().split()
            cleaned_content = [float(item.strip("[],")) for item in lines if item.strip("[],")]
            weighted_MADS_SOO_outputs_arr = np.array(cleaned_content).reshape(-1, 3)
            
            # Use regular expressions to extract the numbers from each line
            Weighted_MADS_HP_filename = f"weighted_MADS_HP_candidates_{weight_idx}.txt"
            with open(Weighted_MADS_HP_filename, "r") as file:
                lines = file.read().split()
            cleaned_content = [float(item.strip("[],")) for item in lines if item.strip("[],")]
            weighted_MADS_HP_arr = np.array(cleaned_content).reshape(-1, 3)
     
            weight_idx_arr = np.full((weighted_MADS_SOO_outputs_arr.shape[0], 1), weight_idx)
            MADS_results_arr = np.column_stack((weight_idx_arr, weighted_MADS_SOO_outputs_arr, weighted_MADS_HP_arr))
            
            # Save the array to a text file
            paths='./MADS_results/ResNet/'
            Weighted_MADS_merge_file_filename = f"{weight_idx}_weighted_MADS_FuncVal_HP_{next_record_time}.txt"
            Weighted_MADS_merge_file = os.path.join(paths, Weighted_MADS_merge_file_filename)
            np.savetxt(Weighted_MADS_merge_file, MADS_results_arr, fmt='%.4f', delimiter=' ')
            
            next_record_time += 60  # Set the next recording time
            
        # Check if the time budget has been exceeded
        if MADS_computational_time_overall > time_budget:
            print("Time budget exceeded. Stopping execution.")
            break
        
        iteration += 1
    return x_init_list


# ------ BO in joint method ------
def single_obj_optimization_EI(x_init):
    dpd_score = []
    density = []
    mse = []
    mbe = []
    accuracy_error = []
    
    if x_init.shape[0] != 1:
        for i in range(x_init.shape[0]):
            y_pred, training_time_list, ResNet_trained_model = NN_prediction(x_init[i], test_features, train_loader, val_loader, task_type=task_type)
            
            # Save the model to disk
            model_filename = f"ResNet_trained_model_{i}.pth"  # The filename includes the iteration index
            
            if os.path.exists(model_filename):
                os.remove(model_filename)
            
            torch.save({'model_state_dict': ResNet_trained_model.state_dict()}, model_filename)
            
            # Objectives for regression
            #new_mse = mean_squared_error_NOMAD(y_pred, test_targets)
            #mse.append(new_mse)
            
            #new_mbe = mean_bias_error_NOMAD(y_pred, test_targets)
            #mbe.append(new_mbe)
            
            #new_sparsity = sparsity_NOMAD(ResNet_trained_model)
            #density.append(new_sparsity)
            
            # Objectives for multi-class classification
            new_accuracy = accuracy_NOMAD(y_pred, test_targets)
            accuracy_error.append(new_accuracy)

            new_sparsity = sparsity_NOMAD(ResNet_trained_model)
            density.append(new_sparsity)
            
            new_dpd = dpd_score_NOMAD(y_pred, test_targets, group=X_test[fairness_feature].values)
            dpd_score.append(new_dpd)
        
        # Objectives for regression
        #mse_arr = np.round(np.array(mse), decimals=4).astype(np.float64)
        #mbe_arr = np.round(np.array(mbe), decimals=4).astype(np.float64)
        #density_arr = np.round(np.array(density), decimals=4).astype(np.float64)
        
        # Objectives for multi-class classification
        accerror_arr = np.round(np.array(accuracy_error), decimals=4).astype(np.float64)  
        density_arr = np.round(np.array(density), decimals=4).astype(np.float64)    
        dpd_arr = np.round(np.array(dpd_score), decimals=4).astype(np.float64)
        
        sum_objectives_array = (weight_1 * accerror_arr + weight_2 * density_arr + weight_3 * dpd_arr).reshape(-1, 1)
        combined_new_array = np.column_stack((accerror_arr, density_arr, dpd_arr))
            
        # write the HP configs into file
        paths='.'
        jm_y_data_file = os.path.join(paths, 'joint_method_y_data.txt')
        if os.path.exists('joint_method_y_data.txt'):
            os.remove('joint_method_y_data.txt') 
        np.savetxt(jm_y_data_file, sum_objectives_array, fmt='%.4f', delimiter=' ')

    if x_init.shape[0] == ndoe or x_init.shape[0] == 1:
        if x_init.shape[0] == 1:
            x_init = x_init.reshape(x_init.shape[1],)

            y_pred, training_time_list, ResNet_trained_model = NN_prediction(x_init, test_features, train_loader, val_loader, task_type=task_type)
            
            # Read the content of the txt file
            joint_method_y_data_path = 'joint_method_y_data.txt'
            with open(joint_method_y_data_path, "r") as file:
                lines = file.read().split()
            
            # Save the model to disk
            model_filename = f"ResNet_trained_model_{len(lines)}.pth"
            
            if os.path.exists(model_filename):
                os.remove(model_filename)
            
            torch.save({'model_state_dict': ResNet_trained_model.state_dict()}, model_filename)
            
            # Objectives for regression
            #new_mse = mean_squared_error_NOMAD(y_pred, test_targets)
            #mse.append(new_mse)
            
            #new_mbe = mean_bias_error_NOMAD(y_pred, test_targets)
            #mbe.append(new_mbe)
            
            #new_sparsity = sparsity_NOMAD(ResNet_trained_model)
            #density.append(new_sparsity)
            
            # Objectives for multi-class classification
            new_accuracy = accuracy_NOMAD(y_pred, test_targets)
            accuracy_error.append(new_accuracy)
            
            new_sparsity = sparsity_NOMAD(ResNet_trained_model)
            density.append(new_sparsity)
            
            new_dpd = dpd_score_NOMAD(y_pred, test_targets, group=X_test[fairness_feature].values)
            dpd_score.append(new_dpd)
            
            # Objectives for regression
            #mse_arr = np.round(np.array(mse), decimals=4).astype(np.float64)
            #mbe_arr = np.round(np.array(mbe), decimals=4).astype(np.float64)
            #dpd_arr = np.round(np.array(dpd_score), decimals=4).astype(np.float64)
            
            # Objectives for multi-class classification
            accerror_arr = np.round(np.array(accuracy_error), decimals=4).astype(np.float64)
            density_arr = np.round(np.array(density), decimals=4).astype(np.float64)    
            dpd_arr = np.round(np.array(dpd_score), decimals=4).astype(np.float64)

            sum_objectives_array = (weight_1 * accerror_arr + weight_2 * density_arr + weight_3 * dpd_arr).reshape(-1, 1)
            combined_new_array = np.column_stack((accerror_arr, density_arr, dpd_arr))
        
        # write the HP configs into file
        paths='.'
        joint_method_outputs_filename = f"joint_method_outputs_{weight_idx}.txt"
        joint_method_outputs_file = open(os.path.join(paths, joint_method_outputs_filename), "a")
        joint_method_outputs_file.write(str(combined_new_array)+'\n' )
        joint_method_outputs_file.close()
        
        joint_method_HP_filename = f"joint_method_HP_{weight_idx}.txt"
        HP_candidates_file = open(os.path.join(paths, joint_method_HP_filename), "a")
        HP_candidates_file.write(str(x_init)+'\n' )
        HP_candidates_file.close()   
    return sum_objectives_array


# ------ MADS in joint method ------
def bb(x):
    try:
        # Read the content of the txt file
        jm_BO_HP_file_path = 'Optimal_HP_BO_to_MADS.txt'
        
        with open(jm_BO_HP_file_path, "r") as file:
            lines = file.read().split()

        cleaned_content = [float(item.strip("[],")) for item in lines if item.strip("[],")]
        hp_configs = np.array(cleaned_content).reshape(-1, 4)
        
        x_init = hp_configs[-1]

        if isinstance(x_init, np.ndarray):
        # Convert NumPy array to list
            x_init = x_init.tolist()
        else:
            # If input is already a list, keep it as is
            x_init = x_init
        
        x_init[:2] = np.round(x_init[:2]).astype(int)
        x_init[-1] = np.round(x_init[-1]).astype(int)  # The index of the trained model

        
        dim = x.size()
        x_new = [x.get_coord(i) for i in range(dim)]
        x_new[:2] = np.round(x_new[:2]).astype(int)
        
        # Warm start
        new_net = change_resnet_structure_ver3(x_new, x_init, task_type=task_type).to(dtype=dtype, device=device)
        y_pred = new_net(test_features)
        
        if task_type == 'multiclass':
            class_indices = y_pred.argmax(dim=1)
            y_pred = torch.nn.functional.one_hot(class_indices, num_classes=y_pred.shape[1])
        
        # Objectives for regression
        #mse = mean_squared_error_NOMAD(y_pred, test_targets)   
        #mbe = mean_bias_error_NOMAD(y_pred, test_targets) 
        #density = sparsity_NOMAD(new_net)
        
        # Objectives for multi-class classification
        accuracy_error = accuracy_NOMAD(y_pred, test_targets) 
        density = sparsity_NOMAD(new_net)
        dpd = dpd_score_NOMAD(y_pred, test_targets, group=X_test[fairness_feature].values)
        
        sum_objectives = weight_1 * accuracy_error + weight_2 * density + weight_3 * dpd
        combined_new_array = np.column_stack((accuracy_error, density, dpd))
        
        x_new = np.array(x_new).reshape(1, -1)

        # write the HP configs into file
        paths='.'
        joint_method_HP_filename = f"joint_method_HP_{weight_idx}.txt"
        HP_candidates_file = open(os.path.join(paths, joint_method_HP_filename), "a")
        HP_candidates_file.write(str(x_new)+'\n' )
        HP_candidates_file.close()
        
        # write the output values into file
        paths='.'
        joint_method_outputs_filename = f"joint_method_outputs_{weight_idx}.txt"        
        joint_method_outputs_file = open(os.path.join(paths, joint_method_outputs_filename), "a")
        joint_method_outputs_file.write(str(combined_new_array)+'\n' )
        joint_method_outputs_file.close()

        x.setBBO(str(sum_objectives).encode("UTF-8"))
    except:
        print("Unexpected eval error", sys.exc_info()[0])
        return 0
    return 1 # 1: success 0: failed evaluation


# ------ Perform the BO in the joint method ------
def joint_weighted_BO(x_init): 
    ego = EGO(
    n_iter = n_iter,
    criterion = criterion,
    xdoe = x_init,
    surrogate = sm,
    n_start = 30)  #to do multistart for maximizing the acquisition function

    x_opt, y_opt, ind_best, x_data = ego.optimize(fun = single_obj_optimization_EI)
    
    # Append the matching line number to the target_row array
    target_row_with_line = np.append(x_opt, ind_best)
    return target_row_with_line, y_opt, x_data

# ------ Perform the MADS in the joint method ------
def joint_weighted_MADS(x_init):
    x_init = x_init[:3]
    x_init[:2] = np.round(x_init[:2]).astype(int) 
    if  x_init[2] < 0.01:
        x_init[2] = 0.01

    if  x_init[2] > 0.8:
        x_init[2] = 0.8
        
    if isinstance(x_init, np.ndarray):
        # Convert NumPy array to list
        x_init = x_init.tolist()
    else:
        # If input is already a list, keep it as is
        x_init = x_init
            
    params = ["DIMENSION 3", "BB_OUTPUT_TYPE OBJ", "MAX_BB_EVAL 6", "DISPLAY_DEGREE 2", "DISPLAY_ALL_EVAL false", "DISPLAY_STATS BBE OBJ",
    "DIRECTION_TYPE N+1 UNI"] 

    result = PyNomad.optimize(bb, x_init, lb, ub, params)
    x_opt = result['x_best']
    obj_best = result['f_best']

    return x_opt, obj_best

# --------- Perform joint method -------------
def run_joint_method(x_init, max_iterations=1000, time_budget=600):  
    global current_method

    iteration = 0
    MADS_fail_iteration = 0
    current_method = joint_weighted_BO   # Start with BO
    obj_opt_list = []
    x_init_list =[]
    
    next_record_time = 60  # Set the first recording time to 30 seconds
    start_time_overall = time.time()
    while iteration < max_iterations:
        print('iteration', iteration)
        start_time = time.time()

        if current_method == joint_weighted_BO:
            x_opt, y_opt, x_all_data = current_method(x_init)  # run BO
            x_init = np.vstack((x_opt[:-1], x_all_data[-3:,:]))
            BO_x_optimal = x_init
            obj_opt_list.append(y_opt)
            x_init_list.extend(x_opt[:-1])

            paths='.'
            jm_BO_HP_file = open(os.path.join(paths, 'Optimal_HP_BO_to_MADS.txt'), "a")
            jm_BO_HP_file.write(str(x_opt)+'\n' )
            jm_BO_HP_file.close()

        elif current_method == joint_weighted_MADS:
            x_opt, obj_opt = current_method(x_init)  # run MADS
            x_init = x_opt
            obj_opt_list.append(obj_opt)
            x_init_list.extend(x_opt)
            
        computational_time_overall = time.time() - start_time_overall
        computational_time = time.time() - start_time
        
        # Switching logic
        if current_method == joint_weighted_BO:
            # Switch from BO to MADS: if the computational time is too long seconds AND not obtain sufficient descent (i.e. the difference between the last two objective values < 1e-2)
            if (computational_time > 2) and (len(obj_opt_list) == 1 or (obj_opt_list[-2] - obj_opt_list[-1]) < 1e-2) and (MADS_fail_iteration <= 60): 
                current_method = joint_weighted_MADS
                x_init = x_opt
        
        elif current_method == joint_weighted_MADS: 
            # Switch from MADS to BO: not obtain sufficient descent (i.e. the difference between the last two objective values > 1e-2) OR 
            # MADS fails to find a better solution for certain times
            if ((len(obj_opt_list) == 1) or (obj_opt_list[-2] - obj_opt_list[-1]) < 1e-2) or (MADS_fail_iteration <= 60): 
                current_method = joint_weighted_BO
                MADS_fail_iteration += 1
                x_opt_arr = np.array(x_opt).reshape(1, -1)
                x_init = np.vstack((BO_x_optimal, x_opt_arr))
        
        # Check if it's time to record the result
        if computational_time_overall >= next_record_time:                
            joint_method_outputs_filename = f"joint_method_outputs_{weight_idx}.txt"
            with open(joint_method_outputs_filename, "r") as file:
                lines = file.read().split()
            cleaned_content = [float(item.strip("[],")) for item in lines if item.strip("[],")]
            joint_method_outputs_arr = np.array(cleaned_content).reshape(-1, 3)
            
            
            joint_method_HP_filename = f"joint_method_HP_{weight_idx}.txt"
            with open(joint_method_HP_filename, "r") as file:
                lines = file.read().split()
            cleaned_content = [float(item.strip("[],")) for item in lines if item.strip("[],")]
            joint_method_HP_arr = np.array(cleaned_content).reshape(-1, 3)
            
            weight_idx_arr = np.full((joint_method_outputs_arr.shape[0], 1), weight_idx)
            joint_method_results_arr = np.column_stack((weight_idx_arr, joint_method_outputs_arr, joint_method_HP_arr))
            
            # Save the array to a text file
            paths='./Joint_method_results/ResNet/'
            joint_method_merge_file_filename = f"{weight_idx}_joint_method_FuncVal_HP_{next_record_time}.txt"
            joint_method_merge_file = os.path.join(paths, joint_method_merge_file_filename)
            np.savetxt(joint_method_merge_file, joint_method_results_arr, fmt='%.4f', delimiter=' ')
            
            next_record_time += 60  # Set the next recording time

        # Check if the time budget has been exceeded
        if computational_time_overall > time_budget:
            print("Time budget exceeded. Stopping execution.")
            break

        iteration += 1
    return x_init_list


time_budget = 600
max_iterations = 100000
n_iter = 1

# Choose different sets of weights
for weights in sets_of_weights:
    weight_idx = find_index(weights)
    weight_1, weight_2, weight_3 = weights
    print('weights', (weight_1, weight_2, weight_3))
    
    x_init = sampling(ndoe)
    
    # ------ Perform joint method ------
    joint_method_all_HP_configs = run_joint_method(x_init, max_iterations=max_iterations, time_budget=time_budget)

    # ------ Perform weighted BO ------
    weighted_BO_all_HP_configs = run_weighted_BO(x_init, max_iterations=max_iterations, time_budget=time_budget)
    
    # ------ Perform weighted MADS ------
    x_init = [max_number_res_block/2, max_neurons_per_layer/2, max_dropout_rate/2]
    weighted_MADS_all_HP_configs = run_weighted_MADS(x_init, max_iterations=max_iterations, time_budget=time_budget)
    
    os.remove('joint_method_y_data.txt')
    os.remove('Optimal_HP_BO_to_MADS.txt')
    
    joint_method_outputs_filename = f"joint_method_outputs_{weight_idx}.txt"
    os.remove(joint_method_outputs_filename)
    
    joint_method_HP_filename = f"joint_method_HP_{weight_idx}.txt"
    os.remove(joint_method_HP_filename)
    
    Weighted_BO_outputs_filename = f"weighted_BO_SOO_outputs_{weight_idx}.txt"
    os.remove(Weighted_BO_outputs_filename)
    
    weighted_BO_HP_filename = f"weighted_BO_HP_{weight_idx}.txt"
    os.remove(weighted_BO_HP_filename)
    
    Weighted_MADS_outputs_filename = f"weighted_MADS_SOO_outputs_{weight_idx}.txt"
    os.remove(Weighted_MADS_outputs_filename)
    
    Weighted_MADS_HP_filename = f"weighted_MADS_HP_candidates_{weight_idx}.txt"
    os.remove(Weighted_MADS_HP_filename)
    
    
# Compute the hypervolume    
reference_point = np.array([1+1e-7, 1+1e-7, 1+1e-7])
time_interval = [60, 120, 180, 240, 300, 360, 420, 480, 540, 600]

hvs_joint_method = []
hvs_BO = []
hvs_MADS = []

Joint_method_all_results_list = []
BO_all_results_list = []
MADS_all_results_list = []

    
for j in time_interval:
    for i in range(len(sets_of_weights)):
        joint_method_results_filename_path = f"Joint_method_results/ResNet/{i}_joint_method_FuncVal_HP_{j}.txt"
        BO_results_filename_path = f"BO_results/ResNet/{i}_weighted_BO_FuncVal_HP_{j}.txt"
        MADS_results_filename_path = f"MADS_results/ResNet/{i}_weighted_MADS_FuncVal_HP_{j}.txt"
        
        with open(joint_method_results_filename_path, "r") as file:
            lines = file.read().split()
        cleaned_content = [float(item.strip("[],")) for item in lines if item.strip("[],")]
        Joint_method_all_results_arr = np.array(cleaned_content).reshape(-1, 7)

        with open(BO_results_filename_path, "r") as file:
            lines = file.read().split()
        cleaned_content = [float(item.strip("[],")) for item in lines if item.strip("[],")]
        BO_all_results_arr = np.array(cleaned_content).reshape(-1, 7)
        
        with open(MADS_results_filename_path, "r") as file:
            lines = file.read().split()
        cleaned_content = [float(item.strip("[],")) for item in lines if item.strip("[],")]
        MADS_all_results_arr = np.array(cleaned_content).reshape(-1, 7)
        
        if task_type == 'regression':
            # Normalize MSE and MBE, but not density 
            max_MADS_values = np.max(MADS_all_results_arr[:, 1:3], axis=0)
            max_BO_values = np.max(BO_all_results_arr[:, 1:3], axis=0)
            max_joint_values = np.max(Joint_method_all_results_arr[:, 1:3], axis=0)
            max_value = np.max(np.vstack([max_MADS_values, max_BO_values, max_joint_values]), axis=0)

            # Normalize the first two columns by maximum(max(MADS), max(BO), max(joint))
            MADS_normalized_first_two_columns = MADS_all_results_arr[:, 1:3] / max_value
            BO_normalized_first_two_columns = BO_all_results_arr[:, 1:3] / max_value
            joint_normalized_first_two_columns = Joint_method_all_results_arr[:, 1:3] / max_value

            # Concatenate the unchanged last column (i.e. density) to the normalized columns
            MADS_normalized_arr = np.concatenate((MADS_normalized_first_two_columns, MADS_all_results_arr[:, 3:4]), axis=1)
            BO_normalized_arr = np.concatenate((BO_normalized_first_two_columns, BO_all_results_arr[:, 3:4]), axis=1)
            joint_normalized_arr = np.concatenate((joint_normalized_first_two_columns, Joint_method_all_results_arr[:, 3:4]), axis=1)
            
            MADS_all_results_list.append(MADS_normalized_arr.tolist())
            BO_all_results_list.append(BO_normalized_arr.tolist())
            Joint_method_all_results_list.append(joint_normalized_arr.tolist())
        
        elif task_type == 'multiclass':
            # Metrics are accuracy error, density, and DPD. Their values are all between 0 and 1, so no normalize is needed!
            MADS_all_results_list.append(MADS_all_results_arr[:, 1:4])
            BO_all_results_list.append(BO_all_results_arr[:, 1:4])
            Joint_method_all_results_list.append(Joint_method_all_results_arr[:, 1:4])
            
    MADS_all_results_array = np.vstack(MADS_all_results_list)
    BO_all_results_array = np.vstack(BO_all_results_list) 
    Joint_method_all_results_array = np.vstack(Joint_method_all_results_list)  
    
    hyp_val = hypervolume(MADS_all_results_array).compute(reference_point)
    hyp_val = np.round(hyp_val,decimals=6).astype(float)
    hvs_MADS.append(hyp_val)  
    
    hyp_val = hypervolume(BO_all_results_array).compute(reference_point)
    hyp_val = np.round(hyp_val,decimals=6).astype(float)
    hvs_BO.append(hyp_val)
    
    hyp_val = hypervolume(Joint_method_all_results_array).compute(reference_point)
    hyp_val = np.round(hyp_val,decimals=6).astype(float)
    hvs_joint_method.append(hyp_val)


# Plot the hypervolume vs. computational time
fig, ax = plt.subplots(figsize=(7, 7))

# Weighted BO 
ax.scatter(time_interval, hvs_BO, marker='*') 
ax.plot(time_interval, hvs_BO, label='Weighted BO (ResNet)', marker='*', linestyle='-.')

# Weighted MADS 
ax.scatter(time_interval, hvs_MADS, marker='D')
ax.plot(time_interval, hvs_MADS, label='Weighted MADS (ResNet)', marker='D', linestyle='--')

# Joint method 
ax.scatter(time_interval, hvs_joint_method, marker='o')
ax.plot(time_interval, hvs_joint_method, label='Joint method (ResNet)', marker='o', linestyle='-')

ax.set_xlabel('Computational time (seconds)', fontsize=11)
ax.set_ylabel('Hypervolume', fontsize=11 )
ax.grid(True, which='both', linestyle='-', linewidth=0.5)
ax.legend(loc='lower right', fontsize=10)
plt.show()
