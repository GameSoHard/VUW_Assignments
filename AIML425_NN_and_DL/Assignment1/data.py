"""
data.py

This module provides functionality for generating and handling synthetic data for
training and testing machine learning models.

Functions:
    - map_one_point: Maps a single N-dimensional data point to the N-dimensional sphere surface.
    - multivariate_norm_generator: Generates synthetic data following a multivariate normal
    distribution.
    - multinorm_to_sphere_mapper: Maps a set of multinormal data points to the sphere surface.

Classes:
    - SphereSurfaceDataset: A PyTorch Dataset class for handling multinorm data points
    and their corresponding sphere surface coordinates.

Usage:
    1. Import the functions and classes from this module to generate and preprocess data.
    2. Use `multivariate_norm_generator` to generate synthetic multinormal data.
    3. Use `multinorm_to_sphere_mapper` to convert the generated data to the sphere surface.
    4. Create an instance of `SphereSurfaceDataset` to handle the data for training and 
    validation in PyTorch.

Example:
    from data import multivariate_norm_generator, multinorm_to_sphere_mapper, SphereSurfaceDataset

    # Generate synthetic data
    multinorm_data = multivariate_norm_generator(MU, COV_MAT, SAMPLE_SIZE)
    sphere_coord = multinorm_to_sphere_mapper(multinorm_data)

    # Create a dataset instance
    dataset = SphereSurfaceDataset(multinorm_data, sphere_coord)
    print(len(dataset))
"""

from typing import List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ['MU', 'COV_MAT', 'SAMPLE_SIZE', 'map_one_point', 'multivariate_norm_generator',
           'multinorm_to_sphere_mapper', 'SphereSurfaceDataset']

MU = [0, 0, 0]
COV_MAT = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
SAMPLE_SIZE = 500


def map_one_point(datapoint_in: List[float]) -> np.ndarray:
    """Map one N-dimensional data point to N-dimensional sphere surface.

    Args:
        datapoint_in (List[float]): An N-dimensional data point.

    Returns:
        np.ndarray: Mapped point on the N-dimensional sphere surface."""
    return datapoint_in / np.linalg.norm(datapoint_in)


def multivariate_norm_generator(mu_in: List[float], cov_mat_in: List[List[float]],
                                sample_size_in: int, set_seed: Optional[int] = None) -> np.ndarray:
    """Generate dataset of size sample_size_in following N-dimensional Gaussian
    distribution with given mean and covariance matrix.

    Args:
        mu_in (List[float]): Mean of the multivariate normal distribution.
        cov_mat_in (List[List[float]]): Covariance matrix of the multivariate normal distribution.
        sample_size_in (int): Number of samples to generate.
        set_seed (Optional[int]): Whether to set a random seed for extra randomness. 
        Defaults to None.

    Returns:
        np.ndarray: Generated dataset."""
    
    if set_seed:
        np.random.seed(set_seed)
        
    return np.random.multivariate_normal(mu_in, cov_mat_in, sample_size_in)


def multinorm_to_sphere_mapper(multinorm_in):
    """Map all multinorm datapoints to sphere surface by Euclidean distance scaling.

    Args:
        multinorm_in (np.ndarray): Array of N-dimensional datapoints.

    Returns:
        np.ndarray: Array of points mapped onto the N-dimensional sphere surface."""
    return np.array(list(map(map_one_point, multinorm_in)))


class SphereSurfaceDataset(Dataset):
    """Dataset of multinorm data points and corresponding coordinates mapped to unit sphere surface.

   Parent class: Dataset (torch.utils.data.Dataset): Abstract base class,
   the __len__ and __getitem__ methods need to be overriden. 
    """
    
    def __init__(self, multinorm_array_in: np.ndarray, sphere_coord_in: np.ndarray):
        """SphereSurfaceDataset class initialization

        Args:
            multinorm_array_in (np.ndarray): generated multinorm datapoints.
            sphere_coord_in (np.ndarray): unit sphere surface coordinates mapped
            from multinorm datapoints.
        """
        assert multinorm_array_in.shape[0] == \
            sphere_coord_in.shape[0], 'Input and out put dimensionalites do not match.'
        
        self.input = multinorm_array_in
        self.ground_truth = sphere_coord_in


    def __len__(self) -> int:
        """Get the size of dataset.

        Returns:
            int: the size of input dataset.
        """
        return self.input.shape[0]
    
    def __getitem__(self, idx_in: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a data point from dataset with given idex.

        Args:
            idx_in (int): index of the data point to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: tuple containing rhe input datapoint 
            and ground truth.
        """
        input_value = torch.tensor(self.input[idx_in], dtype=torch.float32)
        mapped_value = torch.tensor(self.ground_truth[idx_in], dtype=torch.float32)
        
        return input_value, mapped_value



if __name__ == "__main__":

    multinorm_data = multivariate_norm_generator(MU, COV_MAT, SAMPLE_SIZE)
    sphere_coord = multinorm_to_sphere_mapper(multinorm_data)

    print(sphere_coord)
