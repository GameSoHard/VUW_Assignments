"""
This module contains functions and scripts to evaluate a trained neural network model on a test dataset.

Functions:
- model_test: Computes the average loss and MSE on the test dataset using a specified loss function, and records outputs and targets for error analysis.
- Main script execution: Loads the trained model, evaluates its performance on the test dataset, and prints the results.

Usage:
1. Load the test data and the trained model.
2. Use the `model_test` function to calculate the average test loss and MSE, and obtain the recorded outputs and targets.
3. Print the results to assess model performance.

"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from data import SphereSurfaceDataset, multivariate_norm_generator, multinorm_to_sphere_mapper, MU, COV_MAT
from model import SimpleMLP

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def model_test(model_in, test_loader_in, criterion_in):
    """
    Evaluate the model's performance on the test dataset.

    Args:
        model (nn.Module): The trained model to be evaluated.
        test_loader (DataLoader): DataLoader for the test data.
        criterion (nn.Module): Loss function used for evaluation.

    Returns:
        tuple: A tuple containing:
            - float: The average loss on the test dataset.
            - float: The average Mean Squared Error (MSE) on the test dataset.
            - DataFrame: A DataFrame recording the outputs and targets for error analysis.
    """
    model_in.eval()
    test_loss = 0.0
    squared_errors = 0.0
    total_samples = 0
    results = {'outputs': [], 'targets': [], 'distances': [], 'vector_length': []}

    with torch.no_grad():
        for inputs, targets in test_loader_in:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model_in(inputs)
            loss = criterion_in(outputs, targets)
            test_loss += loss.item()
            squared_errors += ((outputs - targets) ** 2).sum().item()
            total_samples += targets.size(0)

            outputs_np = outputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            distances = np.linalg.norm(outputs_np - targets_np, axis=1)
            norms = np.linalg.norm(outputs_np, axis=1)

            results['outputs'].extend(outputs_np)
            results['targets'].extend(targets_np)
            results['distances'].extend(distances)
            results['vector_length'].extend(norms)

    avg_test_loss_res = test_loss / len(test_loader_in)
    avg_mse_res = squared_errors / total_samples
    results_df = pd.DataFrame(results)
    return avg_test_loss_res, avg_mse_res, results_df

if __name__ == "__main__":
    # Parameters
    BATCH_SIZE = 64
    TEST_SIZE = 2000
    MODEL_PATH = './best_model.pth'

    # Load test data
    test_multinorm = multivariate_norm_generator(MU, COV_MAT, TEST_SIZE, 32167)
    test_sphere = multinorm_to_sphere_mapper(test_multinorm)
    test_dataset = SphereSurfaceDataset(test_multinorm, test_sphere)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load the best model
    model = SimpleMLP().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))

    # Evaluation
    criterion = torch.nn.SmoothL1Loss()
    avg_test_loss, avg_mse, df_result = model_test(model, test_loader, criterion)

    print(f'Average Test Loss (SmoothL1): {avg_test_loss}')
    print(f'Average Test MSE: {avg_mse}')
    
    # Save the results for error analysis
    df_result.to_csv('error_analysis_results.csv', index=False)
    print("Saved outputs and targets for error analysis to 'error_analysis_results.csv'")