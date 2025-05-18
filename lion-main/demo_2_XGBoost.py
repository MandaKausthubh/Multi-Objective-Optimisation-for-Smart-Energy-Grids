# The focus of our research is multi-objective optimization in machine learning, 
# i.e., we aim to minimize multiple objectives simultaneously. 
# This demo shows the performance of weighted BO, weighted MADS, and joint method on a XGBoost model. 
# For the regression problem, the objectives are MSE, MBE, and averge depth. 
# For multi-class classification, the objectives are accuracy error, averge depth, and DPD. 
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

import xgboost as xgb

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
target_column = 'TransnetBW..MW.'   # 'X50Hertz..MW.' / 'TransnetBW..MW.'

# Multi-class classification
features = data.drop(columns=[target_column])

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler to the feature subset and transform it
scaled_features = scaler.fit_transform(features)

# Convert the scaled features back to a DataFrame with column names
scaled_features_df = pd.DataFrame(scaled_features, columns=features.columns)
scaled_data = pd.concat([scaled_features_df, data[target_column]], axis=1)

# Define the number of classes
num_classes = 5

# Use qcut to divide the target column into quantile-based bins and assign integer labels
class_labels = pd.qcut(data[target_column], q=num_classes, labels=False)

# Your X and y for model training
X = data.drop(columns=[target_column])
y = class_labels


# ------
# data
# ------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42) 
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state = 42) 

fairness_feature = 'GHI'   # fairness_feature is only used in multi-class classification

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
dvalid = xgb.DMatrix(X_val, label=y_val)


# Hyperparameters for the model
dim = 5

min_n_estimators = 1
max_n_estimators = 50

min_max_depth = 2
max_max_depth = 20

min_max_leaves = 1
max_max_leaves = 20

min_min_child_weight = 0.1
max_min_child_weight = 5   

min_gamma = 0
max_gamma = 0.5

# Parameter setting for BO (SOO)
# xlimits: float (continuous) variables: array-like of size nx x 2 (i.e. lower, upper bounds)
xlimits = np.array([[min_n_estimators, max_n_estimators], 
                    [min_max_depth, max_max_depth],
                    [min_max_leaves, max_max_leaves],
                    [min_min_child_weight, max_min_child_weight],
                    [min_gamma, max_gamma]])

criterion='EI' #'EI' or 'SBO' or 'LCB'


#number of points in the initial x_init
ndoe = 10 #(at least ndim+1)

#design_space = DesignSpace(xlimits)
design_space = DesignSpace ([
    IntegerVariable (min_n_estimators, max_n_estimators),
    IntegerVariable (min_max_depth, max_max_depth), 
    IntegerVariable (min_max_leaves, max_max_leaves),
    FloatVariable (min_min_child_weight, max_min_child_weight),
    FloatVariable (min_gamma, max_gamma)])


#Build the initial DOE, add the random_state option to have the reproducibility of the LHS points
sampling = LHS(xlimits=xlimits, random_state=42)
#x_init = sampling(ndoe)

#EGO call
sm = KRG(design_space=design_space, n_start=25, print_global=False)


# lower boundary & upper boundary
lb = [min_n_estimators, min_max_depth, min_max_leaves, min_min_child_weight, min_gamma]
ub = [max_n_estimators, max_max_depth, max_max_leaves, max_min_child_weight, max_gamma]


reference_point = np.array([1+1e-7, 1+1e-7, 1+1e-7])
hvs_weighted_BO_SOO, hvs_weighted_MADS_SOO, hvs_joint_method_SOO = [], [], []
computational_time_BO_SOO, computational_time_MADS_SOO, computational_time_joint_method_SOO = [], [], []


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
    dpd_score_list = []
    mse_list = []
    mbe_list = []
    acc_list = []
    avg_depth_list = []

    # Fixed parameters
    fixed_params = {
            'eta': 2,   # Learning rate
            'objective': 'multi:softmax',   # 'reg:squarederror'  or  'multi:softmax'  
            'num_class': 5}    

    for i in range(x_init.shape[0]):
        # Initial values for other parameters
        other_params = {'n_estimators': np.round(x_init[i][0]).astype(int),
                        'max_depth': np.round(x_init[i][1]).astype(int),
                        'max_leaves': np.round(x_init[i][2]).astype(int),
                        'min_child_weight': x_init[i][3] if x_init[i][3] > min_min_child_weight else min_min_child_weight,
                        'gamma': x_init[i][4] if x_init[i][4] > min_gamma else min_gamma}
        
        # Combine fixed and other parameters for initial training
        params = {**fixed_params, **other_params}
        num_rounds = params['n_estimators']

        if x_init.shape[0] == ndoe:
            # Save the model to disk
            xgb_model_filename = f"prev_xgb_model_{i}.model"  
            if os.path.exists(xgb_model_filename):
                os.remove(xgb_model_filename)
            xgb_model = xgb.train(params, dtrain, num_boost_round = num_rounds, 
                                evals=[(dvalid, 'eval')], early_stopping_rounds=3)
            prev_xgb_model = xgb_model
            prev_xgb_model.save_model(xgb_model_filename)
            
        elif (x_init.shape[0] == 6) or (x_init.shape[0] == 7):
            BO_opt_idx_file_path = 'optimal_xgb_idx.txt'
            with open(BO_opt_idx_file_path, "r") as file:
                    lines = file.read().split()
            cleaned_content = [float(item.strip("[],")) for item in lines if item.strip("[],")]
            optimal_model_idx = np.round(cleaned_content[-1]).astype(int)
            
            xgb_model_filename = f"prev_xgb_model_{optimal_model_idx}.model"
            print('xgb_model_filename', xgb_model_filename)
            prev_xgb_model = xgb.Booster()
            prev_xgb_model.load_model(xgb_model_filename)
            
            xgb_model = xgb.train(params, dtrain, num_boost_round = num_rounds, 
                                evals=[(dvalid, 'eval')], early_stopping_rounds=3, xgb_model=prev_xgb_model)
            prev_xgb_model = xgb_model

            
        elif x_init.shape[0] == 1:
            # Read the content of the txt file
            BO_opt_idx_file_path = 'optimal_xgb_idx.txt'
            if os.path.exists(BO_opt_idx_file_path):     
                with open(BO_opt_idx_file_path, "r") as file:
                    lines = file.read().split()
                cleaned_content = [float(item.strip("[],")) for item in lines if item.strip("[],")]
                optimal_model_idx = np.round(cleaned_content[-1]).astype(int)
                
                xgb_model_filename = f"prev_xgb_model_{optimal_model_idx}.model"
                prev_xgb_model = xgb.Booster()
                prev_xgb_model.load_model(xgb_model_filename)
                
                xgb_model = xgb.train(params, dtrain, num_boost_round = num_rounds, 
                                    evals=[(dvalid, 'eval')], early_stopping_rounds=3, xgb_model=prev_xgb_model)
                prev_xgb_model = xgb_model
                os.remove(xgb_model_filename)
                prev_xgb_model.save_model(xgb_model_filename)
            
            else:
                if os.path.exists('prev_xgb_model_10.model'):
                    os.remove('prev_xgb_model_10.model')
                xgb_model = xgb.train(params, dtrain, num_boost_round = num_rounds, 
                                    evals=[(dvalid, 'eval')], early_stopping_rounds=3)
                prev_xgb_model = xgb_model
                prev_xgb_model.save_model('prev_xgb_model_10.model')

        predictions = prev_xgb_model.predict(dtest)
        
        # Objectives for regression
        #mse = mean_squared_error_xgb(predictions, y_test.values)
        #mse_list.append(mse)
        
        #mbe = mean_bias_error_xgb(predictions, y_test.values)
        #mbe_list.append(mbe)
        
        #avg_depth = average_depth_xgb(prev_xgb_model)
        #avg_depth_list.append(avg_depth)
        
        # Objectives for classification
        accuracy_error = (1- sklearn.metrics.accuracy_score(y_test, predictions))
        acc_list.append(accuracy_error)
        
        dpd = dpd_score_xgb(predictions, y_test, group=X_test[fairness_feature])
        dpd_score_list.append(dpd)
        
        avg_depth = average_depth_xgb(prev_xgb_model)
        avg_depth_list.append(avg_depth)
    
    # Objectives for regression
    #mse_arr = np.round(np.array(mse_list), decimals=4).astype(np.float64)
    #mbe_arr = np.round(np.array(mbe_list), decimals=4).astype(np.float64)
    #avg_depth_arr = np.round(np.array(avg_depth_list), decimals=4).astype(np.float64)
    
    # Objectives for classification
    acc_arr = np.round(np.array(acc_list), decimals=4).astype(np.float64)   
    avg_depth_arr = np.round(np.array(avg_depth_list), decimals=4).astype(np.float64)
    dpd_arr = np.round(np.array(dpd_score_list), decimals=4).astype(np.float64)

    sum_objectives_array = (weight_1 * acc_arr + weight_2 * avg_depth_arr + weight_3 * dpd_arr).reshape(-1, 1) 
    combined_new_array = np.column_stack((acc_arr, avg_depth_arr, dpd_arr))
        
    if x_init.shape[0] == ndoe or x_init.shape[0] == 1:
        # write the HP configs into file
        paths='.'
        Weighted_BO_SOO_outputs_filename = f"weighted_BO_SOO_outputs_{weight_idx}.txt"
        Weighted_BO_SOO_outputs_file = open(os.path.join(paths, Weighted_BO_SOO_outputs_filename), "a")
        Weighted_BO_SOO_outputs_file.write(str(combined_new_array)+'\n' )
        Weighted_BO_SOO_outputs_file.close()
        
        Weighted_BO_HP_filename = f"weighted_BO_HP_{weight_idx}.txt"      
        HP_candidates_file = open(os.path.join(paths, Weighted_BO_HP_filename), "a")
        HP_candidates_file.write(str(x_init)+'\n' )
        HP_candidates_file.close()
    
    return sum_objectives_array

# ------ Perform weighted BO ------
def run_weighted_BO(x_init, max_iterations=1000, time_budget=30):
    iteration = 0 
    x_init_list = []  # List to store the results and times
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
        #x_init = np.tile(x_opt, (4, 1))
        x_init = np.vstack((x_opt, x_data[-5:,:]))
        x_init_list.extend(x_opt)
        
        paths='.'
        BO_idx_file = open(os.path.join(paths, 'optimal_xgb_idx.txt'), "a")
        BO_idx_file.write(str(ind_best)+'\n' )
        BO_idx_file.close()
        
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
            weighted_BO_HP_arr = np.array(cleaned_content).reshape(-1, 5)
            
            weight_idx_arr = np.full((weighted_BO_outputs_arr.shape[0], 1), weight_idx)
            BO_results_arr = np.column_stack((weight_idx_arr, weighted_BO_outputs_arr, weighted_BO_HP_arr))
            
            # Save the array to a text file
            paths='./BO_results/XGBoost/'
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
        x_init[:3] = np.round(x_init[:3]).astype(int)

        # Fixed parameters
        fixed_params = {
            'eta': 2,   # Learning rate
            'objective': 'multi:softmax',   # 'reg:squarederror'  or  'multi:softmax'  
            'num_class': 5}
            
        other_params = {'n_estimators': x_init[0],
                        'max_depth': x_init[1],
                        'max_leaves': x_init[2],
                        'min_child_weight': x_init[3],
                        'gamma': x_init[4]}
        
        # Combine fixed and other parameters for initial training
        params = {**fixed_params, **other_params}
        num_rounds = other_params['n_estimators']
        xgb_model_filename = f"prev_xgb_model.model" 
            
        if os.path.exists(xgb_model_filename):    
            # Load the initial model if it's saved
            prev_xgb_model = xgb.Booster()
            prev_xgb_model.load_model(xgb_model_filename)

            # Continue training with the updated parameters            
            xgb_model = xgb.train(params, dtrain, num_boost_round = num_rounds, 
                                evals=[(dvalid, 'eval')], early_stopping_rounds=3, xgb_model=prev_xgb_model)
            prev_xgb_model = xgb_model
            os.remove(xgb_model_filename)
            # Save the model
            prev_xgb_model.save_model(xgb_model_filename)
        else:
            xgb_model = xgb.train(params, dtrain, num_boost_round = num_rounds, 
                                evals=[(dvalid, 'eval')], early_stopping_rounds=3)
            prev_xgb_model = xgb_model
            prev_xgb_model.save_model(xgb_model_filename)
        
        predictions = prev_xgb_model.predict(dtest)

        # Objectives for regression
        #mse = mean_squared_error_xgb(predictions, y_test.values)
        #mbe = mean_bias_error_xgb(predictions, y_test.values)
        #avg_depth = average_depth_xgb(prev_xgb_model)
        
        # Objectives for classification
        accuracy_error = (1 - sklearn.metrics.accuracy_score(y_test, predictions))
        dpd = dpd_score_xgb(predictions, y_test, group=X_test[fairness_feature])
        avg_depth = average_depth_xgb(prev_xgb_model)
        
        sum_objectives = weight_1 * accuracy_error + weight_2 * avg_depth + weight_3 * dpd
        combined_new_array = np.column_stack((accuracy_error, avg_depth, dpd))
        x_init_arr = np.array(x_init).reshape(1, -1)
        
        # write the HP configs into file
        paths='.'
        Weighted_MADS_HP_candidates_file_filename = f"weighted_MADS_HP_candidates_{weight_idx}.txt"
        HP_candidates_file = open(os.path.join(paths, Weighted_MADS_HP_candidates_file_filename), "a")
        HP_candidates_file.write(str(x_init_arr)+'\n' )
        HP_candidates_file.close()
        
        # write the objective values into file
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
        params_only_MADS = ["DIMENSION 5", "BB_OUTPUT_TYPE OBJ", "MAX_BB_EVAL 10", "DISPLAY_DEGREE 2", "DISPLAY_ALL_EVAL false", "DISPLAY_STATS BBE OBJ",
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

            Weighted_MADS_HP_filename = f"weighted_MADS_HP_candidates_{weight_idx}.txt"    
            with open(Weighted_MADS_HP_filename, "r") as file:
                lines = file.read().split()
            cleaned_content = [float(item.strip("[],")) for item in lines if item.strip("[],")]
            weighted_MADS_HP_arr = np.array(cleaned_content).reshape(-1, 5)
  
            weight_idx_arr = np.full((weighted_MADS_SOO_outputs_arr.shape[0], 1), weight_idx)
            MADS_results_arr = np.column_stack((weight_idx_arr, weighted_MADS_SOO_outputs_arr, weighted_MADS_HP_arr))
            
            paths='./MADS_results/XGBoost/'
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
    if x_init.shape[0] != 1:
        paths='.'
        BO_input_shape_file = open(os.path.join(paths, 'xgb_BO_input_shape.txt'), "a")
        BO_input_shape_file.write(str(x_init.shape[0])+'\n' )
        BO_input_shape_file.close()
    
    dpd_score_list = []
    mse_list = []
    mbe_list = []
    acc_list = []
    avg_depth_list = []
    
    # Fixed parameters
    fixed_params = {
            'eta': 2,   # Learning rate
            'objective': 'multi:softmax' ,    # 'reg:squarederror'  or  'multi:softmax'  
            'num_class': 5}    
    
    for i in range(x_init.shape[0]):
        # Initial values for other parameters
        other_params = {'n_estimators': np.round(x_init[i][0]).astype(int),
                        'max_depth': np.round(x_init[i][1]).astype(int),
                        'max_leaves': np.round(x_init[i][2]).astype(int),
                        'min_child_weight': x_init[i][3] if x_init[i][3] > min_min_child_weight else min_min_child_weight,
                        'gamma': x_init[i][4] if x_init[i][4] > min_gamma else min_gamma}
        
        # Combine fixed and other parameters for initial training
        params = {**fixed_params, **other_params}
        num_rounds = params['n_estimators']

        if x_init.shape[0] == ndoe:
            # Save the model to disk
            xgb_model_filename = f"prev_xgb_model_{i}.model"  # The filename includes the iteration index
            if os.path.exists(xgb_model_filename):
                os.remove(xgb_model_filename)
            xgb_model = xgb.train(params, dtrain, num_boost_round = num_rounds, 
                                evals=[(dvalid, 'eval')], early_stopping_rounds=3)
            prev_xgb_model = xgb_model
            prev_xgb_model.save_model(xgb_model_filename)
            
        elif (x_init.shape[0] == 6) or (x_init.shape[0] == 7):
            BO_opt_idx_file_path = 'optimal_xgb_idx.txt'
            with open(BO_opt_idx_file_path, "r") as file:
                    lines = file.read().split()
            cleaned_content = [float(item.strip("[],")) for item in lines if item.strip("[],")]
            optimal_model_idx = np.round(cleaned_content[-1]).astype(int)
            
            xgb_model_filename = f"prev_xgb_model_{optimal_model_idx}.model"
            prev_xgb_model = xgb.Booster()
            prev_xgb_model.load_model(xgb_model_filename)
            
            xgb_model = xgb.train(params, dtrain, num_boost_round = num_rounds, 
                                evals=[(dvalid, 'eval')], early_stopping_rounds=3, xgb_model=prev_xgb_model)
            prev_xgb_model = xgb_model
            
            if i == optimal_model_idx:
                i = optimal_model_idx+1
            new_xgb_model_filename = f"prev_xgb_model_{i}.model"
            os.remove(new_xgb_model_filename)
            prev_xgb_model.save_model(new_xgb_model_filename)
        
        elif x_init.shape[0] == 1:
            # Read the content of the txt file
            BO_opt_idx_file_path = 'optimal_xgb_idx.txt'
            if os.path.exists(BO_opt_idx_file_path):     
                with open(BO_opt_idx_file_path, "r") as file:
                    lines = file.read().split()
                cleaned_content = [float(item.strip("[],")) for item in lines if item.strip("[],")]
                
                optimal_model_idx = np.round(cleaned_content[-1]).astype(int)
                
                xgb_model_filename = f"prev_xgb_model_{optimal_model_idx}.model"
                print('xgb_model_filename', xgb_model_filename)
                prev_xgb_model = xgb.Booster()
                prev_xgb_model.load_model(xgb_model_filename)
                
                xgb_model = xgb.train(params, dtrain, num_boost_round = num_rounds, 
                                    evals=[(dvalid, 'eval')], early_stopping_rounds=3, xgb_model=prev_xgb_model)
                prev_xgb_model = xgb_model
                with open('xgb_BO_input_shape.txt', "r") as file:
                    lines = file.read().split()
                cleaned_content = [float(item.strip("[],")) for item in lines if item.strip("[],")]
                
                BO_input_shape = np.round(cleaned_content[-1]).astype(int)
                new_xgb_model_filename = f"prev_xgb_model_{BO_input_shape}.model"
                os.remove(new_xgb_model_filename)
                prev_xgb_model.save_model(new_xgb_model_filename)
            else:
                if os.path.exists('prev_xgb_model_10.model'):
                    os.remove('prev_xgb_model_10.model')
                xgb_model = xgb.train(params, dtrain, num_boost_round = num_rounds, 
                                    evals=[(dvalid, 'eval')], early_stopping_rounds=3)
                prev_xgb_model = xgb_model
                prev_xgb_model.save_model('prev_xgb_model_10.model')
        
        predictions = prev_xgb_model.predict(dtest)
        
        # Objectives for regression
        #mse = mean_squared_error_xgb(predictions, y_test.values)
        #mse_list.append(mse)
        
        #mbe = mean_bias_error_xgb(predictions, y_test.values)
        #mbe_list.append(mbe)
        
        #avg_depth = average_depth_xgb(prev_xgb_model)
        #avg_depth_list.append(avg_depth)
        
        # Objectives for classification
        accuracy_error = (1- sklearn.metrics.accuracy_score(y_test, predictions))
        acc_list.append(accuracy_error)
        
        dpd = dpd_score_xgb(predictions, y_test, group=X_test[fairness_feature])
        dpd_score_list.append(dpd)
        
        avg_depth = average_depth_xgb(prev_xgb_model)
        avg_depth_list.append(avg_depth)
    
    # Objectives for regression
    #mse_arr = np.round(np.array(mse_list), decimals=4).astype(np.float64)
    #mbe_arr = np.round(np.array(mbe_list), decimals=4).astype(np.float64)
    #avg_depth_arr = np.round(np.array(avg_depth_list), decimals=4).astype(np.float64)
    
    # Objectives for classification
    acc_arr = np.round(np.array(acc_list), decimals=4).astype(np.float64)   
    avg_depth_arr = np.round(np.array(avg_depth_list), decimals=4).astype(np.float64)
    dpd_arr = np.round(np.array(dpd_score_list), decimals=4).astype(np.float64)

    sum_objectives_array = (weight_1 * acc_arr + weight_2 * avg_depth_arr + weight_3 * dpd_arr).reshape(-1, 1) 
    combined_new_array = np.column_stack((acc_arr, avg_depth_arr, dpd_arr))
    
    if x_init.shape[0] == ndoe or x_init.shape[0] == 1:
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
        dim = x.size()
        x_init = [x.get_coord(i)  for i in range(dim)]
        x_init[:3] = np.round(x_init[:3]).astype(int)

        # Fixed parameters
        fixed_params = {
            'eta': 2,   # Learning rate
            'objective': 'multi:softmax' ,   # 'reg:squarederror'  or  'multi:softmax'  
            'num_class': 5}

        if x_init[0] > max_n_estimators:
            x_init[0] = max_n_estimators
        elif x_init[0] < min_n_estimators:
            x_init[0] = min_n_estimators
        
        if x_init[1] > max_max_depth:
            x_init[1] = max_max_depth
        elif x_init[1] < min_max_depth:
            x_init[1] = min_max_depth
        
        if x_init[2] > max_max_leaves:
            x_init[2] = max_max_leaves
        elif x_init[2] < min_max_leaves:
            x_init[2] = min_max_leaves
        
        if x_init[3] > max_min_child_weight:
            x_init[3] = max_min_child_weight
        elif x_init[3] < min_min_child_weight:
            x_init[3] = min_min_child_weight
        
        if x_init[4] > max_gamma:
            x_init[4] = max_gamma
        elif x_init[4] < min_gamma:
            x_init[4] = min_gamma
            
        other_params = {'n_estimators': x_init[0],
                        'max_depth': x_init[1],
                        'max_leaves': x_init[2],
                        'min_child_weight': x_init[3],
                        'gamma': x_init[4]}
        
        # Combine fixed and other parameters for initial training
        params = {**fixed_params, **other_params}
        num_rounds = other_params['n_estimators']
        
        # Read the content of the txt file
        BO_opt_idx_file_path = 'optimal_xgb_idx.txt'   
        with open(BO_opt_idx_file_path, "r") as file:
            lines = file.read().split()
        cleaned_content = [float(item.strip("[],")) for item in lines if item.strip("[],")]

        optimal_model_idx = np.round(cleaned_content[-1]).astype(int)
        
        xgb_model_filename = f"prev_xgb_model_{optimal_model_idx}.model"
        prev_xgb_model = xgb.Booster()
        prev_xgb_model.load_model(xgb_model_filename)
        
        xgb_model = xgb.train(params, dtrain, num_boost_round = num_rounds, 
                            evals=[(dvalid, 'eval')], early_stopping_rounds=3, xgb_model=prev_xgb_model)
        prev_xgb_model = xgb_model
        os.remove(xgb_model_filename)
        prev_xgb_model.save_model(xgb_model_filename)
        
        predictions = prev_xgb_model.predict(dtest)

        # Objectives for regression
        #mse = mean_squared_error_xgb(predictions, y_test.values)
        #mbe = mean_bias_error_xgb(predictions, y_test.values)
        #avg_depth = average_depth_xgb(prev_xgb_model)
        
        # Objectives for classification
        accuracy_error = (1 - sklearn.metrics.accuracy_score(y_test, predictions))
        dpd = dpd_score_xgb(predictions, y_test, group=X_test[fairness_feature])
        avg_depth = average_depth_xgb(prev_xgb_model)

        sum_objectives = weight_1 * accuracy_error + weight_2 * avg_depth + weight_3 * dpd
        combined_new_array = np.column_stack((accuracy_error, avg_depth, dpd))
        x_init_arr = np.array(x_init).reshape(1, -1)
    
        paths='.'
        joint_method_HP_filename = f"joint_method_HP_{weight_idx}.txt"
        HP_candidates_file = open(os.path.join(paths, joint_method_HP_filename), "a")
        HP_candidates_file.write(str(x_init_arr)+'\n' )
        HP_candidates_file.close()
          
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

        x_opt, y_opt, ind_best, x_data, y_data = ego.optimize(fun = single_obj_optimization_EI)
        
        paths='.'
        BO_idx_file = open(os.path.join(paths, 'optimal_xgb_idx.txt'), "a")
        BO_idx_file.write(str(ind_best)+'\n' )
        BO_idx_file.close()
        
        return x_opt, y_opt, x_data, y_data

# ------ Perform the MADS in the joint method ------
def joint_weighted_MADS(x_init):
    if x_init[0] > max_n_estimators:
        x_init[0] = max_n_estimators
    elif x_init[0] < min_n_estimators:
        x_init[0] = min_n_estimators
    
    if x_init[1] > max_max_depth:
        x_init[1] = max_max_depth
    elif x_init[1] < min_max_depth:
        x_init[1] = min_max_depth
    
    if x_init[2] > max_max_leaves:
        x_init[2] = max_max_leaves
    elif x_init[2] < min_max_leaves:
        x_init[2] = min_max_leaves
    
    if x_init[3] > max_min_child_weight:
        x_init[3] = max_min_child_weight
    elif x_init[3] < min_min_child_weight:
        x_init[3] = min_min_child_weight
    
    if x_init[4] > max_gamma:
        x_init[4] = max_gamma
    elif x_init[4] < min_gamma:
        x_init[4] = min_gamma
        
    if isinstance(x_init, np.ndarray):
        # Convert NumPy array to list
        x_init = x_init.tolist()
    else:
        # If input is already a list, keep it as is
        x_init = x_init

        
    params = ["DIMENSION 5", "BB_OUTPUT_TYPE OBJ", "MAX_BB_EVAL 10", "DISPLAY_DEGREE 2", "DISPLAY_ALL_EVAL false", "DISPLAY_STATS BBE OBJ",
    "DIRECTION_TYPE N+1 UNI"]
    
    result = PyNomad.optimize(bb, x_init, lb, ub, params)
    x_opt = result['x_best']
    obj_best = result['f_best']

    return x_opt, obj_best

# --------- Perform joint method -------------
def run_joint_method(x_init, max_iterations=1000, time_budget=30): 
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
            x_opt, y_opt, x_all_data, y_all_data = current_method(x_init)  # run BO
            #x_init = np.tile(x_opt, (6, 1))
            x_init = np.vstack((x_opt, x_all_data[-5:,:])) 
            BO_x_optimal = x_init
            obj_opt_list.append(y_opt)
            x_init_list.extend(x_opt)
        
        elif current_method == joint_weighted_MADS:
            x_opt, obj_opt = current_method(x_init)  # run MADS
            x_init = x_opt
            obj_opt_list.append(obj_opt)
            x_init_list.extend(x_opt)
            
        computational_time_overall = time.time() - start_time_overall
        computational_time = time.time() - start_time
        #print('computational_time', computational_time)
        
        # Switching logic
        if current_method == joint_weighted_BO:
            # Switch from BO to MADS: if the computational time is too long seconds AND not obtain sufficient descent (i.e. the difference between the last two objective values < 1e-2)
            if (computational_time > 0.5) and (len(obj_opt_list) == 1 or (obj_opt_list[-2] - obj_opt_list[-1]) < 1e-2) and (MADS_fail_iteration <= 60): 
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
            joint_method_HP_arr = np.array(cleaned_content).reshape(-1, 5)
            
            weight_idx_arr = np.full((joint_method_outputs_arr.shape[0], 1), weight_idx)
            joint_method_results_arr = np.column_stack((weight_idx_arr, joint_method_outputs_arr, joint_method_HP_arr))
            
            # Save the array to a text file
            paths='./Joint_method_results/XGBoost/'
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
    x_init = [max_n_estimators/2, max_max_depth/2, max_max_leaves/2, max_min_child_weight/2, max_gamma/2]
    weighted_MADS_all_HP_configs = run_weighted_MADS(x_init, max_iterations=max_iterations, time_budget=time_budget)
    
    os.remove('optimal_xgb_idx.txt')
    os.remove('xgb_BO_input_shape.txt')
    
    joint_method_HP_filename = f"joint_method_HP_{weight_idx}.txt"
    os.remove(joint_method_HP_filename)
    
    joint_method_outputs_filename = f"joint_method_outputs_{weight_idx}.txt"
    os.remove(joint_method_outputs_filename)  
    
    Weighted_BO_outputs_filename = f"weighted_BO_SOO_outputs_{weight_idx}.txt"
    os.remove(Weighted_BO_outputs_filename)
    
    weighted_BO_HP_filename = f"weighted_BO_HP_{weight_idx}.txt"
    os.remove(weighted_BO_HP_filename)
    
    xgb_model_filename = f"prev_xgb_model.model"
    os.remove(xgb_model_filename)     
    
    Weighted_MADS_SOO_outputs_filename = f"weighted_MADS_SOO_outputs_{weight_idx}.txt"
    os.remove(Weighted_MADS_SOO_outputs_filename)
    
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
        joint_method_results_filename_path = f"Joint_method_results/XGBoost/{i}_joint_method_FuncVal_HP_{j}.txt"
        BO_results_filename_path = f"BO_results/XGBoost/{i}_weighted_BO_FuncVal_HP_{j}.txt"
        MADS_results_filename_path = f"MADS_results/XGBoost/{i}_weighted_MADS_FuncVal_HP_{j}.txt"
        
        with open(joint_method_results_filename_path, "r") as file:
            lines = file.read().split()
        cleaned_content = [float(item.strip("[],")) for item in lines if item.strip("[],")]
        Joint_method_all_results_arr = np.array(cleaned_content).reshape(-1, 9)

        with open(BO_results_filename_path, "r") as file:
            lines = file.read().split()
        cleaned_content = [float(item.strip("[],")) for item in lines if item.strip("[],")]
        BO_all_results_arr = np.array(cleaned_content).reshape(-1, 9)
        
        with open(MADS_results_filename_path, "r") as file:
            lines = file.read().split()
        cleaned_content = [float(item.strip("[],")) for item in lines if item.strip("[],")]
        MADS_all_results_arr = np.array(cleaned_content).reshape(-1, 9)
    
        if task_type == 'regression':
            # Normalize MSE, MBE and avg_depth by maximum(max(MADS), max(BO), max(Joint))
            max_MADS_values = np.max(MADS_all_results_arr[:, 1:4], axis=0)
            max_BO_values = np.max(BO_all_results_arr[:, 1:4], axis=0)
            max_joint_values = np.max(Joint_method_all_results_arr[:, 1:4], axis=0)
            max_value = np.max(np.vstack([max_MADS_values, max_BO_values, max_joint_values]), axis=0)

            MADS_normalized_arr = MADS_all_results_arr[:, 1:4] / max_value
            BO_normalized_arr = BO_all_results_arr[:, 1:4] / max_value
            joint_normalized_arr = Joint_method_all_results_arr[:, 1:4] / max_value
            
            MADS_all_results_list.append(MADS_normalized_arr.tolist())
            BO_all_results_list.append(BO_normalized_arr.tolist())
            Joint_method_all_results_list.append(joint_normalized_arr.tolist())
        
        elif task_type == 'multiclass':
            # Only normalize avg_depth (not accuracy_error and dpd)
            max_MADS_values = np.max(MADS_all_results_arr[:, 2:3], axis=0)
            max_BO_values = np.max(BO_all_results_arr[:, 2:3], axis=0)
            max_joint_values = np.max(Joint_method_all_results_arr[:, 2:3], axis=0)
            max_value = np.max(np.vstack([max_MADS_values, max_BO_values, max_joint_values]), axis=0)

            MADS_all_results_arr[:, 2:3] = MADS_all_results_arr[:, 2:3] / max_value
            BO_all_results_arr[:, 2:3] = BO_all_results_arr[:, 2:3] / max_value
            Joint_method_all_results_arr[:, 2:3] = Joint_method_all_results_arr[:, 2:3] / max_value
            
            MADS_all_results_list.append(MADS_all_results_arr[:, 1:4].tolist())
            BO_all_results_list.append(BO_all_results_arr[:, 1:4].tolist())
            Joint_method_all_results_list.append(Joint_method_all_results_arr[:, 1:4].tolist())
        
    Joint_method_all_results_array = np.vstack(Joint_method_all_results_list)    
    BO_all_results_array = np.vstack(BO_all_results_list) 
    MADS_all_results_array = np.vstack(MADS_all_results_list)
    
    hyp_val = hypervolume(Joint_method_all_results_array).compute(reference_point)
    hyp_val = np.round(hyp_val,decimals=6).astype(float)
    hvs_joint_method.append(hyp_val)

    hyp_val = hypervolume(BO_all_results_array).compute(reference_point)
    hyp_val = np.round(hyp_val,decimals=6).astype(float)
    hvs_BO.append(hyp_val)

    hyp_val = hypervolume(MADS_all_results_array).compute(reference_point)
    hyp_val = np.round(hyp_val,decimals=6).astype(float)
    hvs_MADS.append(hyp_val)


# Plot the hypervolume vs. computational time
fig, ax = plt.subplots(figsize=(7, 7))

# Weighted BO (SOO)
ax.scatter(time_interval, hvs_BO, marker='*')
ax.plot(time_interval, hvs_BO, label='Weighted BO (XGBoost)', marker='*', linestyle='-.')

# Weighted MADS (SOO)
ax.scatter(time_interval, hvs_MADS, marker='D')
ax.plot(time_interval, hvs_MADS, label='Weighted MADS (XGBoost)', marker='D', linestyle='--')

# Joint method (SOO)
ax.scatter(time_interval, hvs_joint_method, marker='o')
ax.plot(time_interval, hvs_joint_method, label='Joint method (XGBoost)', marker='o', linestyle='-')

ax.set_xlabel('Computational time (seconds)', fontsize=11)
ax.set_ylabel('Hypervolume', fontsize=11)
ax.grid(True, which='both', linestyle='-', linewidth=0.5)
ax.legend(loc='center right', fontsize=10)
plt.show()