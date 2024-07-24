"""
This module contains functions for training and evaluating a neural network model using PyTorch. It includes:

1. `create_dataloaders`: Splits the raw dataset into training, validation, and test sets, and creates DataLoader instances for each set.
2. `initialize_weights`: Initializes the weights of the model using He initialization.
3. `train_and_evaluate`: Trains the model for a specified number of epochs and evaluates it on the validation set. Records gradients and losses for analysis.

The script also handles the following tasks:
- Loading and preparing the dataset.
- Initializing and configuring the model.
- Training the model and saving the best-performing model based on validation loss.

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import DataLoader, random_split
import pandas as pd
from data import SphereSurfaceDataset, multivariate_norm_generator, multinorm_to_sphere_mapper, MU, COV_MAT
from model import SimpleMLP
from plotting import plot_losses, plot_gradients

TRAIN_RATIO = 0.8
VAL_RATIO = 0.2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_dataloaders(raw_dataset_in, test_size_in=2000, batch_size_in=64, 
                       train_ratio_in=TRAIN_RATIO, val_ratio_in=VAL_RATIO):
    """
    Create DataLoader instances for training, validation, and testing datasets.

    Args:
        raw_dataset_in (Dataset): The raw dataset to be split.
        test_size (int): Number of samples in the test dataset. Default is 2000.
        batch_size_in (int): Batch size for the DataLoader. Default is 64.
        train_ratio_in (float): Ratio of data to be used for training. Default is 0.8.
        val_ratio_in (float): Ratio of data to be used for validation. Default is 0.2.

    Returns:
        tuple: A tuple containing the training DataLoader, validation DataLoader,
        and test DataLoader.
    """
    total_size = len(raw_dataset_in)
    train_size = int(train_ratio_in * total_size)
    val_size = int(val_ratio_in * total_size)

    train_dataset, val_dataset = random_split(raw_dataset_in, [train_size, val_size])
    traindata_loader = DataLoader(train_dataset, batch_size=batch_size_in, shuffle=True)
    valdata_loader = DataLoader(val_dataset, batch_size=batch_size_in, shuffle=False)
    
    test_multinorm = multivariate_norm_generator(MU, COV_MAT, test_size_in, 32167)
    test_sphere = multinorm_to_sphere_mapper(test_multinorm)
    test_dataset = SphereSurfaceDataset(test_multinorm, test_sphere)
    testdata_loader = DataLoader(test_dataset, batch_size=batch_size_in, shuffle=False)
    
    return traindata_loader, valdata_loader, testdata_loader

def initialize_weights(model_in):
    """
    Initialize the weights of the model using He initialization.

    Args:
        model (nn.Module): The model to be initialized.
        init_type (str): Type of initialization. Options are 'he'.
    """
    for curr_module in model_in.modules():
        if isinstance(curr_module, nn.Linear):
            init.kaiming_uniform_(curr_module.weight, nonlinearity='relu')
            if curr_module.bias is not None:
                init.constant_(curr_module.bias, 0)
                
                
def train_and_evaluate(model_in, train_loader_in, val_loader_in, criterion_in,
                       optimizer_in, num_epochs_in=20):
    """
    Train and evaluate the model, recording gradients and losses.

    Args:
        model_in (nn.Module): The model to be trained.
        train_loader_in (DataLoader): DataLoader for the training data.
        val_loader_in (DataLoader): DataLoader for the validation data.
        criterion_in (nn.Module): Loss function.
        optimizer_in (torch.optim.Optimizer): Optimizer for model training.
        num_epochs_in (int): Number of epochs for training. Default is 20.

    Returns:
        tuple: A tuple containing:
            - grad_record (DataFrame): DataFrame with gradient information recorded.
            - train_losses_record (list): List of average training losses per epoch.
            - val_losses_record (list): List of average validation losses per epoch.
    """
    grad_record = pd.DataFrame(columns=['epoch', 'iteration', 'fc1_grad',
                                        'fc2_grad', 'fc3_grad', 'fc4_grad'])
    train_losses_record = []
    val_losses_record = []
    
    best_val_loss = float(10e6)
    
    for epoch in range(num_epochs_in):
        model_in.train()
        running_loss = 0.0
        num_batches = len(train_loader_in)

        for i, (inputs, targets) in enumerate(train_loader_in):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer_in.zero_grad()
            outputs = model_in(inputs)
            loss = criterion_in(outputs, targets)
            loss.backward()

            grad_info = {
                'epoch': epoch + 1,
                'iteration': i + 1,
                'fc1_grad': model_in.fc1.weight.grad.mean().item(),
                'fc2_grad': model_in.fc2.weight.grad.mean().item(),
                'fc3_grad': model_in.fc3.weight.grad.mean().item(),
                'fc4_grad': model_in.fc4.weight.grad.mean().item()
            }
            grad_record = pd.concat([grad_record, pd.DataFrame([grad_info])], ignore_index=True)

            optimizer_in.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / num_batches
        train_losses_record.append(avg_train_loss)
        print(f'Epoch [{epoch+1}/{num_epochs_in}], Loss: {avg_train_loss}')

        model_in.eval()
        with torch.no_grad():
            val_loss = 0.0
            for val_inputs, val_targets in val_loader_in:
                val_inputs, val_targets = val_inputs.to(DEVICE), val_targets.to(DEVICE)
                val_outputs = model_in(val_inputs)
                loss = criterion_in(val_outputs, val_targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader_in)
        val_losses_record.append(avg_val_loss)
        print(f'Validation Loss: {avg_val_loss}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model_in.state_dict(), './best_model.pth')


    return grad_record, train_losses_record, val_losses_record

if __name__ == "__main__":

    # Parameter configuration
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_EPOCHS = 20
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    
    # Loading data
    raw_multinorm = multivariate_norm_generator(MU, COV_MAT, 10000)
    raw_sphere = multinorm_to_sphere_mapper(raw_multinorm)
    raw_dataset = SphereSurfaceDataset(raw_multinorm, raw_sphere)
    train_loader, val_loader, test_loader = create_dataloaders(raw_dataset, batch_size_in=BATCH_SIZE)
    
    # Model and corresponding NN parameters initialization
    MODEL = SimpleMLP().to(DEVICE)
    initialize_weights(MODEL)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)
    
    # Training and validation
    grad_df, train_losses, val_losses = train_and_evaluate(MODEL, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS)
    
    # 绘制损失和梯度变化
    plot_losses(train_losses, val_losses)
    plot_gradients(grad_df, 0.2)
