import os
import scipy.io
import numpy as np
import torch
from typing import Dict, List, Any, Union, Tuple

class MATLABDataReader:
    def __init__(self, directory: str, filename: str) -> None:
        """
        Initialize the MATLABDataReader object by appending the current working directory
        to the specified directory and filename of the .mat file.
        :param directory: Relative path to the directory containing the .mat file.
        :param filename: Name of the .mat file.
        """
        current_working_directory: str = os.getcwd()  # Get current working directory
        full_directory: str = os.path.join(current_working_directory, directory)  # Append the specified directory
        self.filepath: str = os.path.join(full_directory, filename)  # Complete path to the file

        self.stats_dir: str = os.path.join(full_directory, 'preprocessing')
        os.makedirs(self.stats_dir, exist_ok=True)  # Create the preprocessing directory if it doesn't exist

        self.variable_shapes: Dict[str, Any] = {}
        self.data: Dict[str, Any] = self.load_data_all()

    def load_data_all(self) -> Dict[str, Any]:
        """
        Load data from the .mat file.
        :return: A dictionary containing all the data loaded from the .mat file.
        """
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"The file does not exist at the path {self.filepath}.")
        self.data = scipy.io.loadmat(self.filepath)

        for key, value in self.data.items():
            if isinstance(value, np.ndarray):
                self.variable_shapes[key] = value.shape
        return self.data

    def load_data(self) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Load data from the .mat file, extract 'a_smooth' and 'u' as PyTorch tensors,
        and create a grid based on the shape of 'u'.
        :return: A dictionary containing the 'grid', 'a_smooth', and 'u' data as PyTorch tensors.
        """
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"The file does not exist at the path {self.filepath}.")
        self.data = scipy.io.loadmat(self.filepath)
        a_smooth: torch.Tensor = torch.tensor(self.data['a_smooth'], dtype=torch.float)
        u: torch.Tensor = torch.tensor(self.data['u'], dtype=torch.float)
        u = u.unsqueeze(-1)
        grid: torch.Tensor = self.get_grid(u.shape)
        self.variable_shapes['grid'] = grid.shape

        return {'grid': grid, 'a_smooth': a_smooth, 'u': u}

    def get_grid(self, shape: Tuple[int, int, int]) -> torch.Tensor:
        """
        Create a coordinate grid corresponding to the 'u' variable.
        :param shape: Tuple indicating the shape of the tensor for which the grid is generated.
        :return: Tensor representing the grid coordinates.
        """
        batchsize: int = shape[0]
        size_x: int = shape[1]
        gridx: torch.Tensor = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx  # By default, this will be on CPU

    def compute_global_stats(self, data: torch.Tensor, key: str) -> Dict[str, torch.Tensor]:
        """
        Compute mean, standard deviation, min, and max for each feature across all batches.
        This handles data with [batch, feature] and [batch, seq, feature] dimensions.
        :param data: The data for which to compute statistics.
        :param key: The key associated with the data for saving statistics.
        :return: A dictionary containing the computed statistics.
        """
        feature_dim: int = data.shape[-1]  # Last dimension is always treated as feature dimension
        batch_dim: int = 0
        if len(data.shape) > 2:
            data = data.reshape(-1, feature_dim)

        # Compute statistics along the feature dimension
        mean: torch.Tensor = data.mean(dim=batch_dim)
        std: torch.Tensor = data.std(dim=batch_dim)
        min_val: torch.Tensor = data.min(dim=batch_dim).values
        max_val: torch.Tensor = data.max(dim=batch_dim).values

        stats: Dict[str, torch.Tensor] = {'mean': mean, 'std': std, 'min': min_val, 'max': max_val}
        torch.save(stats, os.path.join(self.stats_dir, f'{key}_stats.pth'))

        return stats

    def standardize_data(self, tensor: torch.Tensor, stats: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Standardize each feature in the tensor.
        :param tensor: The tensor to be standardized.
        :param stats: The statistics dictionary containing mean and standard deviation.
        :return: The standardized tensor.
        """
        return (tensor - stats['mean']) / stats['std']

    def normalize_data(self, tensor: torch.Tensor, stats: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Normalize each feature in the tensor.
        :param tensor: The tensor to be normalized.
        :param stats: The statistics dictionary containing min and max values.
        :return: The normalized tensor.
        """
        return (tensor - stats['min']) / (stats['max'] - stats['min'])

    def list_variables(self) -> List[str]:
        """
        List all variables stored in the .mat file.
        :return: List of variable names.
        """
        if not self.data:
            self.load_data()
        return list(self.data.keys())

    def get_variable(self, variable_name: str) -> Any:
        """
        Get a specific variable from the .mat file.
        :param variable_name: Name of the variable to retrieve.
        :return: Data associated with the variable.
        """
        if not self.data:
            self.load_data()
        if variable_name in self.data:
            return self.data[variable_name]
        else:
            raise ValueError(f"Variable '{variable_name}' not found in the file.")

    def print_variable_shapes(self) -> None:
        """
        Print the shape of each variable stored in the .mat file.
        """
        if not self.data:
            self.load_data()
        for variable in self.list_variables():
            # Exclude '__header__', '__version__', and '__globals__' as they are not actual data variables
            if variable not in ['__header__', '__version__', '__globals__']:
                variable_data = self.data[variable]
                print(f"Variable '{variable}': Shape {variable_data.shape}")

# Example usage:
# Create an instance of the reader
# directory = './data/burgers'
# filename = 'burgers_data_R10.mat'
# mat_reader = MATLABDataReader(directory, filename)

# # Load the data
# data = mat_reader.load_data()

# # List variables
# variables = mat_reader.list_variables()
# print("Variables in the file:", variables)

# # Print the shape of each variable
# mat_reader.print_variable_shapes()

# Get a specific variable
# variable_data = mat_reader.get_variable('some_variable_name')
# print("Data for the variable:", variable_data)
