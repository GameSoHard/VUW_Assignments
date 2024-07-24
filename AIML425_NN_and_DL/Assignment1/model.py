"""
model.py

This module defines the architecture of a simple multi-layer perceptron (MLP) neural network.

Classes:
    - SimpleMLP: Defines a simple multi-layer perceptron with three hidden layers 
        and ReLU activations.

Usage:
    1. Import the SimpleMLP class from this module.
    2. Instantiate the SimpleMLP class to create a model.
    3. Use the model for training or inference with your data.

Example:
    from model import SimpleMLP

    # Create a model instance
    model = SimpleMLP()

    # Forward pass with dummy data
    import torch
    test_input = torch.randn(5, 3)  # Batch of 5 samples with 3 features
    output = model(test_input)
    print(output)
"""

import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    """The structure of neural network required in the assignment. Personally, 
        I select layer from 'nn' if there are variates inside the layer, otherwise, use
        nn.functional. 

    Args:
        nn.Module: PyTorch base class for all neural network modules.
    """
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(3, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, 3)
    
    def forward(self, x):
        """Network forward 

        Args:
            x (torch.Tensor): Input datapoint as torch.Tensor of 3-d

        Returns:
            torch.Tensor: Output as torch.Tensor of 3-d
        """
        # F.relu not nn.relu
        x = F.relu(self.fc1(x))  
        x = F.relu(self.fc2(x))  
        x = F.relu(self.fc3(x))
        # No activasion function at the last layer
        x = self.fc4(x)
        return x
