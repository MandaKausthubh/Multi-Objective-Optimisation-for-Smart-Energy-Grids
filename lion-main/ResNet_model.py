# System Level Items -- paths etc. 
import os
import sys
sys.path.append('../')
from io import StringIO

import torch
import torch.nn as nn
import torch.nn.init as init
import time 
import numpy as np
import random


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.double
    
class ResNet_block(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResNet_block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim),
        )
        
    def forward(self, x):
        identity = x
        x = self.layers(x)
        x += identity  # add the skip connection
        x = nn.ReLU()(x)  # apply ReLU activation
        return x


class ResNet_model_torch(nn.Module):
    def __init__(self, input_dim=24, num_blocks=4, neurons_hidden_dims=256, 
                 output_dim=1, dropout_rate=0.1, task_type=None):  # type -- regression, binary, multiclass, multilabel
        super(ResNet_model_torch, self).__init__()
        self.input_dim = input_dim
        self.neurons_hidden_dims = neurons_hidden_dims
        self.out_dim = output_dim
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        
        if dropout_rate > 1:
            dropout_rate = 0.9
        elif dropout_rate < 0:
            dropout_rate = 1e-5
        else:
            pass  # no action needed if 0 <= dropout_rate <= 1

        # create a list to hold the layers
        layers = []
        
        # create the first layer
        layers.append(nn.Linear(self.input_dim, self.neurons_hidden_dims))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout_rate))
        
        # create the residual blocks
        for i in range(self.num_blocks):
            layers.append(ResNet_block(self.neurons_hidden_dims, self.neurons_hidden_dims))
            layers.append(nn.Dropout(self.dropout_rate))
        
        # create the output layer
        layers.append(nn.Linear(self.neurons_hidden_dims, self.out_dim))
        
        if task_type == 'binary': 
            layers.append(nn.Sigmoid())  # Sigmoid activation for binary classification

        # combine all layers into a single module
        self.model = nn.Sequential(*layers)
            
         
    def forward(self, x):
        conv_output = self.model(x)
        return conv_output


class MLP_model_torch(nn.Module):
    def __init__(self, input_dim=24, num_layer=10, neurons_hidden_dims=64, 
                 output_dim=8, dropout_rate=0.1, task_type=None):
        super(MLP_model_torch, self).__init__()
        self.input_dim = input_dim
        self.num_layer = num_layer
        self.neurons_hidden_dims = neurons_hidden_dims
        self.out_dim = output_dim
        
        # create a list to hold the hidden layers
        layers = []
        
        # create the first hidden layer
        layers.append(nn.Linear(self.input_dim, self.neurons_hidden_dims))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # create the remaining hidden layers
        for i in range(self.num_layer):
            layers.append(nn.Linear(self.neurons_hidden_dims, self.neurons_hidden_dims))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        # create the output layer
        layers.append(nn.Linear(self.neurons_hidden_dims, self.out_dim))

        # combine all layers into a single module
        self.model = nn.Sequential(*layers)
            
         
    def forward(self, x):
        conv_output = self.model(x)
        return conv_output



def ResNet_train_model(net, train_loader, valid_loader, num_epochs, learning_rate, dtype, device, patience=5, task_type='multiclass'):
    """
    Trains the network and returns the trained model.
    """
    net.to(dtype = dtype, device = device)
    
    # Define the loss function 
    if task_type == 'regression':
        criterion = torch.nn.MSELoss()   # MSE Loss for regression        
    elif task_type == 'binary':
        criterion = torch.nn.BCELoss()   # Binary Cross-Entropy Loss for binary classification
    elif task_type == 'multiclass':
        criterion = torch.nn.CrossEntropyLoss()  # Cross-Entropy Loss for multi-class classification
    elif task_type == 'multilabel':
        criterion = nn.BCEWithLogitsLoss()   # Combines a Sigmoid layer and the BCELoss for multi-label classification

    # define the optimizer
    #optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.001)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.005)
     
    # initialize variables for early stopping
    best_loss = float('inf')
    counter = 0

    # train the model
    for epoch in range(num_epochs):
        running_loss = 0.0
        #for i, data in enumerate(train_loader, 0):
        for inputs, targets in train_loader:
            # Backward pass and optimization
            optimizer.zero_grad()   # Zero the parameter gradients
            train_outputs = net(inputs)
            loss = criterion(train_outputs, targets)
            loss.backward()   # Perform backpropagation
            optimizer.step()   # Update the weights
            running_loss += loss.item()
        
        # evaluate the model on the validation set
        valid_loss = 0.0
        for inputs, labels in valid_loader:
            with torch.no_grad():                
                vaild_outputs = net(inputs)
                loss = criterion(vaild_outputs, labels)
                valid_loss += loss.item()

        valid_loss /= len(valid_loader)
        
        # check if the validation loss has improved
        if valid_loss < best_loss:
            best_loss = valid_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                #print('Validation loss did not improve for %d epochs, stopping training' % patience)
                break

    return net


def NN_prediction(x, testing_data, training_data, validation_data, training_time=None, task_type='multiclass'):
    x_num_res_block = np.round(x[0]).astype(int)
    neurons_per_layer = np.round(x[1]).astype(int)
    x_dropout = x[2]

    num_epochs = 30
    learning_rate = 1e-2  
    
    training_time_list =[]
                                    
    t0 = time.monotonic()
    ResNet_trained_model = ResNet_train_model(ResNet_model_torch(input_dim=testing_data.shape[1], neurons_hidden_dims=neurons_per_layer, dropout_rate=x_dropout, 
                                    num_blocks=x_num_res_block, output_dim=5), training_data, validation_data, num_epochs, learning_rate, dtype, device, task_type=task_type)
    t1 = time.monotonic()
    training = t1 - t0
    training_time_list.append(training)
    
    with torch.no_grad():  
        y_pred = ResNet_trained_model(testing_data)  # Get the predictions for regression

    if task_type == 'multiclass':
        class_indices = y_pred.argmax(dim=1)
        y_pred = torch.nn.functional.one_hot(class_indices, num_classes=y_pred.shape[1])
  
    return y_pred, training_time_list, ResNet_trained_model



def change_resnet_structure_ver3(next_arch, prev_arch, task_type='regression'):  # Warm start for ResNet

    new_blocks, new_neurons, new_dropout_rate = next_arch
    if len(prev_arch) == 3:
        prev_blocks, prev_neurons, prev_dropout_rate = prev_arch
        x_opt_idx = 0
    elif len(prev_arch) == 4:
        prev_blocks, prev_neurons, prev_dropout_rate, x_opt_idx = prev_arch
    print('next_arch', next_arch)
    print('prev_arch', prev_arch)
    
    if task_type == 'regression':
        output_dim = 1
    elif task_type == 'multiclass':
        output_dim = 5
    
    original_net = ResNet_model_torch(input_dim=23, 
                                      neurons_hidden_dims=prev_neurons, 
                                      dropout_rate=prev_dropout_rate, 
                                      num_blocks=prev_blocks,
                                    output_dim=output_dim, 
                                    task_type=task_type).to(dtype=dtype, device=device)
    
    opt_model_filename = f"ResNet_trained_model_{x_opt_idx}.pth"
    checkpoint = torch.load(opt_model_filename)

    original_net.load_state_dict(checkpoint['model_state_dict'])
            
    if new_blocks == prev_blocks and new_neurons == prev_neurons:
        print('-------------case 1-----------')
        new_net = original_net  

    # If the number of blocks is increasing, add a new block after the last layer
    if prev_blocks < new_blocks:
        print('-------------case 2-----------')
        # Create a new instance of ResNet_model_torch with the updated number of blocks
        new_net = ResNet_model_torch(input_dim=23, 
                                      neurons_hidden_dims=prev_neurons, 
                                      dropout_rate=prev_dropout_rate, 
                                      num_blocks=new_blocks,
                                    output_dim=output_dim, 
                                    task_type=task_type).to(dtype=dtype, device=device)
        # Transfer weights and biases from the original model to the new model
        with torch.no_grad():
            # Get the state dictionaries of both models
            original_state_dict = original_net.state_dict()
            new_state_dict = new_net.state_dict()

            # Update the new model's state dict with weights from the original model
            # Skip the layers that are beyond the original model's length
            for name in original_state_dict:
                if name in new_state_dict and new_state_dict[name].shape == original_state_dict[name].shape:
                    new_state_dict[name].copy_(original_state_dict[name])

        # Load the updated state dict into the new model
        new_net.load_state_dict(new_state_dict)
    
        # If the number of blocks is decreasing, randomly remove a block
    elif prev_blocks > new_blocks:
        print('--------------case 3---------------')
        # Create a new model 
        new_model = ResNet_model_torch(input_dim=23, 
                                      neurons_hidden_dims=prev_neurons, 
                                      dropout_rate=prev_dropout_rate, 
                                      num_blocks=new_blocks,
                                    output_dim=output_dim, 
                                    task_type=task_type).to(dtype=dtype, device=device)

        # Calculate the number of blocks to remove
        num_remove_blocks = prev_blocks - new_blocks
        # Determine which ResNet_block to remove (randomly in this example)
        blocks_to_remove = np.random.choice(prev_blocks, size= num_remove_blocks, replace=False)  # Randomly choose a block to remove
  
        # Extract layers from the original model
        original_layers = list(original_net.model.children())

        # Create a list to hold the new layers
        new_layers = []
        
        # Counter for ResNet blocks
        block_counter = 0

        # Convert blocks_to_remove from a NumPy array to a Python list
        blocks_to_remove = blocks_to_remove.tolist()

        if len(blocks_to_remove) == 1:
            original_state_dict = original_net.state_dict()
            new_state_dict = new_model.state_dict()

            # Initialize a counter for skipping the removed block
            skip_block_counter = 0

            # Transfer weights and biases from the original model to the new model
            with torch.no_grad():
                for name in original_state_dict:
                    # Check if we are at the block to remove
                    if f"{blocks_to_remove[0]}." in name:
                        skip_block_counter = 2  # Skip this block and the following dropout layer
                        continue

                    if skip_block_counter > 0:
                        skip_block_counter -= 1
                        continue

                    # Update the new model's state dict
                    if name in new_state_dict:
                        new_state_dict[name].copy_(original_state_dict[name])
                    #print(f'Copying layer: {name}')

            # Load the updated state dict into the new model
            new_model.load_state_dict(new_state_dict)

        else:   # If more than one block is being removed, i.e. len(blocks_to_remove) > 1
            skip_next = True  # Flag to skip the next layer (Dropout) after a removed ResNet_block
            for layer in original_layers:
                if skip_next:
                    skip_next = False
                    continue  # Skip this layer (Dropout) because the preceding ResNet_block was removed
                if isinstance(layer, nn.Linear):    # if isinstance(layer, ResNet_block): 
                    if block_counter in blocks_to_remove:
                        block_counter += 1
                        skip_next = True  # Set flag to skip the next layer (Dropout)
                        continue  # Skip this block (ResNet_block)
                    block_counter += 1
            
                # Add this layer to the new model
                new_layers.append(layer)

            # Update the new model's layers
            new_model.model = nn.Sequential(*new_layers)

            # Transfer weights
            original_state_dict = original_net.state_dict()
            new_state_dict = new_model.state_dict()
            updated_state_dict = {}
            
            for name, param in original_state_dict.items():
                # Check if this layer should be removed
                #if any(f'model.{2 * block + 4}.' in name for block in blocks_to_remove):
                #    continue  # Skip this layer
                
                # Otherwise, add this layer to the new state dict
                if name in new_state_dict:
                    updated_state_dict[name] = param

            # Update the new model's state dict with the transferred weights
            new_state_dict.update(updated_state_dict)
            
            # Load the new state dictionary into the new model
            new_model.load_state_dict(new_state_dict)

        # return new model,
        # user can now use 'new_model' with the transferred weights and excluded ResNet_blocks        
        new_net = new_model

    elif prev_blocks == new_blocks:
        new_net = original_net 
        
     
    # If the number of neurons is increasing, add a new neuron to each layer    
    if prev_neurons < new_neurons:
        print('-----------case 4--------------')
        num_added_neurons = new_neurons - prev_neurons
        # Define a function to modify the dimensions of linear layers in ResNet blocks
        def add_neuron_to_resnet_model(original_model):
            # Create a new model with an additional neuron in each hidden layer
            new_model = ResNet_model_torch(input_dim=23, 
                                      neurons_hidden_dims=new_neurons, 
                                      dropout_rate=original_model.dropout_rate, 
                                      num_blocks=original_model.num_blocks,
                                    output_dim=output_dim,
                                    task_type=task_type).to(dtype=dtype, device=device)

            # Copy weights and biases from the original model to the new model
            with torch.no_grad():
                for new_layer, original_layer in zip(new_model.model, original_model.model):
                    # Check if both layers are Linear
                    if isinstance(new_layer, nn.Linear) and isinstance(original_layer, nn.Linear):
                        new_layer.weight[:original_layer.weight.shape[0], :original_layer.weight.shape[1]] = original_layer.weight.clone()
                        new_layer.bias[:original_layer.bias.shape[0]] = original_layer.bias.clone()

                        # If it's the output layer, don't set the last row of weights and last bias to zero
                        if new_layer.out_features == original_model.out_dim:
                            continue
                        
                        new_layer.weight[-num_added_neurons:, :].fill_(0.0)
                        new_layer.bias[-num_added_neurons:].fill_(0.0)

                    # Check if both layers are ResNet_block
                    elif isinstance(new_layer, ResNet_block) and isinstance(original_layer, ResNet_block):

                        try:
                            for idx, (new_sub_layer, original_sub_layer) in enumerate(zip(new_layer.layers, original_layer.layers)):
                                #print(f'Inside Sub-layer loop of ResNet_block: Iteration {idx}')

                                new_sub_layer.weight[:original_sub_layer.weight.shape[0], :original_sub_layer.weight.shape[1]] = original_sub_layer.weight.clone()
                                new_sub_layer.bias[:original_sub_layer.bias.shape[0]] = original_sub_layer.bias.clone()
                                new_sub_layer.weight[-num_added_neurons:, :].fill_(0.0)
                                new_sub_layer.bias[-num_added_neurons:].fill_(0.0)
                            #print('Exiting ResNet_block Layer - test 23')
                        except Exception as e:
                            print(f'Exception in sub-layer loop: {e}')
            return new_model

        # Call the function to modify the model
        final_new_net = add_neuron_to_resnet_model(new_net)
    
        # If the number of neurons is decreasing, remove neuron(s) from each layer    
    elif prev_neurons > new_neurons:
        print('---------case 5------------')
        num_remove_neurons = prev_neurons - new_neurons
        def remove_neuron_from_resnet_model(original_model):
            # Create a new model with one fewer neuron in each hidden layer
            new_model = ResNet_model_torch(input_dim=23, 
                                      neurons_hidden_dims=new_neurons, 
                                      dropout_rate=original_model.dropout_rate,
                                      num_blocks=original_model.num_blocks,
                                    output_dim=output_dim,
                                    task_type=task_type).to(dtype=dtype, device=device)

            # Copy weights and biases from the original model to the new model
            with torch.no_grad():
                for new_layer, original_layer in zip(new_model.model, original_model.model):
                    if isinstance(new_layer, nn.Linear) and isinstance(original_layer, nn.Linear):
                        if new_layer.out_features == original_layer.out_features:
                            new_layer.weight.data = original_layer.weight[:, :-num_remove_neurons].data.clone()
                            new_layer.bias.data = original_layer.bias.data.clone()
                            continue

                        remove_indices = random.sample(range(original_layer.weight.shape[0]), num_remove_neurons)
                        keep_indices = [i for i in range(original_layer.weight.shape[0]) if i not in remove_indices]
                        
                        new_layer.weight.data = original_layer.weight.data[keep_indices, :]
                        new_layer.bias.data = original_layer.bias.data[keep_indices]
 
                    elif isinstance(new_layer, ResNet_block) and isinstance(original_layer, ResNet_block):
                        for new_sub_layer, original_sub_layer in zip(new_layer.layers, original_layer.layers):
                            if isinstance(new_sub_layer, nn.Linear) and isinstance(original_sub_layer, nn.Linear):
                                remove_indices = random.sample(range(original_sub_layer.weight.shape[0]), num_remove_neurons)
                                keep_indices = [i for i in range(original_sub_layer.weight.shape[0]) if i not in remove_indices]

                                if new_sub_layer is new_layer.layers[0]:
                                    next_layer = new_layer.layers[2]
                                    next_layer.weight.data = original_layer.layers[2].weight.data[:, keep_indices]
                                
                                new_sub_layer.weight.data = original_sub_layer.weight.data[keep_indices, :-num_remove_neurons]
                                new_sub_layer.bias.data = original_sub_layer.bias.data[keep_indices]

            return new_model
        
        # Create a new model with one fewer neuron in each hidden layer
        final_new_net = remove_neuron_from_resnet_model(new_net)
    
    if prev_neurons == new_neurons:
        return new_net
    else:
        return final_new_net