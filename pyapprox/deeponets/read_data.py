import os
import numpy as np
import torch
from typing import List, Dict, Any

class ReadData:
    def __init__(self, directory: str) -> None:
        """
        Initialize the ReadData object with the directory where the .npz files are saved.
        :param directory: Path to the directory containing the .npz files.
        """
        self.directory: str = directory
        self.preprocessing_dir: str = os.path.join(self.directory, 'preprocessing')
        os.makedirs(self.preprocessing_dir, exist_ok=True)  # Create the preprocessing directory if it doesn't exist

    def load_data(self) -> List[Dict[str, torch.Tensor]]:
        """
        Load all .npz files in the specified directory and convert them into PyTorch tensors.
        :return: A list of dictionaries. Each dictionary contains 'solution', 'coordinates_time', and 'parameter' tensors.
        """
        data_list: List[Dict[str, torch.Tensor]] = []

        # Iterate over each file in the directory
        for filename in os.listdir(self.directory):
            if filename.endswith(".npz"):
                file_path: str = os.path.join(self.directory, filename)
                
                # Extract parameter 'a_val' and time 't' from the filename
                parts: List[str] = filename.split('_')
                a_val: float = float(parts[2])
                t: float = float(parts[4][:-4])

                # Load the .npz file
                data: Dict[str, np.ndarray] = np.load(file_path)

                # Extract 'solution' and 'coordinates', and add 'time' to coordinates
                solution: torch.Tensor = torch.tensor(data['solution'], dtype=torch.float)
                coordinates: torch.Tensor = torch.tensor(data['coordinates'][:, 0:2], dtype=torch.float)
                time: torch.Tensor = torch.full((coordinates.shape[0], 1), t, dtype=torch.float)
                coordinates_time: torch.Tensor = torch.cat((coordinates, time), dim=1)

                # Create a tensor for 'a_val'
                parameter: torch.Tensor = torch.tensor([a_val], dtype=torch.float)

                # Append the tensors to the list
                data_list.append({
                    'solution': solution,
                    'coordinates_time': coordinates_time,
                    'parameter': parameter
                })

        return data_list

    def compute_global_stats(self, data_list: List[Dict[str, torch.Tensor]], key: str) -> Dict[str, torch.Tensor]:
        """
        Compute mean, standard deviation, min, and max for each feature across all batches.
        :param data_list: A list of dictionaries containing the data.
        :param key: The key in the dictionaries for which to compute statistics.
        :return: A dictionary containing the computed statistics.
        """
        all_data: torch.Tensor = torch.cat([item[key] for item in data_list], dim=0)
        mean: torch.Tensor = all_data.mean(dim=0, keepdim=True)
        std: torch.Tensor = all_data.std(dim=0, keepdim=True)
        min_val: torch.Tensor = all_data.min(dim=0, keepdim=True)[0]
        max_val: torch.Tensor = all_data.max(dim=0, keepdim=True)[0]
        self.save_global_stats(mean, std, min_val, max_val, key)

        print(key, all_data.shape, mean.shape, std.shape, min_val.shape, max_val.shape)
        return {'mean': mean, 'std': std, 'min': min_val, 'max': max_val}

    def save_global_stats(self, mean: torch.Tensor, std: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor, key: str) -> None:
        """
        Save the computed global statistics to a file.
        :param mean: The mean tensor.
        :param std: The standard deviation tensor.
        :param min_val: The minimum value tensor.
        :param max_val: The maximum value tensor.
        :param key: The key associated with the data for saving statistics.
        """
        torch.save({'mean': mean, 'std': std, 'min': min_val, 'max': max_val}, os.path.join(self.preprocessing_dir, f'{key}_stats.pth'))

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

# Example usage:
# directory = "simulation_results"
# reader = ReadData(directory)
# data = reader.load_data()
# print(data)
# 'data' now contains the tensors for each file
