import os
import time
import json
import random
from typing import List, Dict, Any
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from read_data import ReadData  # Import the ReadData class
from read_matlab import MATLABDataReader
from networks import DeepONets, DeepONetsWithFNO, DeepONetsWithHybridBranch  # Import the DeepONets class
from networks_torchdiff import DeepONetsWithODEBranch
from networks_fno import DeepONetsWithFNOBranch
from train_utils import train_model, evaluate_model, calculate_relative_test_error, count_parameters
from plot_utils import plot_losses
from save_results_utils import save_losses
from custom_dataset import CustomDataset

def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.
    :param seed: The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main() -> None:
    # Define training parameters
    batch_size: int = 32
    shuffle_dataset: bool = True
    random_seed: int = 42
    num_epochs: int = 150
    learning_rate: float = 0.0005

    # Define dataset directory and filename
    directory: str = './data/burgers'
    filename: str = 'burgers_data_R10.mat'
    mat_reader: MATLABDataReader = MATLABDataReader(directory, filename)

    # Load the data from the MATLAB file
    data: Dict[str, np.ndarray] = mat_reader.load_data()

    # Determine the number of features for branch and trunk inputs
    branch_features: int = mat_reader.variable_shapes['a_smooth'][-1]
    sub_rate: int = 2**5  # Subsampling rate
    branch_features_sub: int = int(branch_features / sub_rate)
    trunk_features: int = mat_reader.variable_shapes['grid'][-1]

    # Compute and save global statistics for normalization and standardization
    for key in ['grid', 'a_smooth', 'u']:
        mat_reader.compute_global_stats(data[key], key)

    # Create a dataset with global normalization and standardization
    full_dataset: CustomDataset = CustomDataset(data, mat_reader, sub_rate)

    # Create data indices for training, validation, and test splits
    dataset_size: int = len(full_dataset)
    indices: List[int] = list(range(dataset_size))
    split_train: int = int(np.floor(0.7 * dataset_size))
    split_val: int = int(np.floor(0.85 * dataset_size))

    # Shuffle the dataset if required
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    # Create training, validation, and test indices
    train_indices: List[int] = indices[:split_train]
    val_indices: List[int] = indices[split_train:split_val]
    test_indices: List[int] = indices[split_val:]

    # Create data samplers and loaders
    train_sampler: SubsetRandomSampler = SubsetRandomSampler(train_indices)
    valid_sampler: SubsetRandomSampler = SubsetRandomSampler(val_indices)
    test_sampler: SubsetRandomSampler = SubsetRandomSampler(test_indices)

    train_loader: DataLoader = DataLoader(full_dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader: DataLoader = DataLoader(full_dataset, batch_size=batch_size, sampler=valid_sampler)
    test_loader: DataLoader = DataLoader(full_dataset, batch_size=batch_size, sampler=test_sampler)

    # Initialize the DeepONets model
    branch_layers: List[int] = [branch_features_sub, 32, 32, 32, 16]
    trunk_layers: List[int] = [trunk_features, 32, 32, 16]

    model: torch.nn.Module = DeepONets(branch_layers=branch_layers, trunk_layers=trunk_layers, out_features=1)
    # Uncomment the relevant model initialization as needed
    # model = DeepONetsWithHybridBranch(branch_layers=branch_layers,
    #                                   trunk_layers=trunk_layers,
    #                                   out_features=1,
    #                                   fourier_features=128)
    # model = DeepONetsWithFNOBranch(branch_layers=branch_layers,
    #                                trunk_layers=trunk_layers,
    #                                out_features=1,
    #                                fourier_features=64,
    #                                modes=16)
    # model = DeepONetsWithODEBranch(in_features, hidden_dim, trunk_layers, out_features)

    # Count the number of parameters in the model
    num_params: int = count_parameters(model)

    # Train the model and save the best one
    save_folder: str = 'DeepONets_burgers'
    start_time: float = time.time()
    train_losses, val_losses = train_model(model, train_loader, validation_loader, save_folder, epochs=num_epochs, learning_rate=learning_rate)
    training_time: float = time.time() - start_time

    # Determine the best epoch based on validation losses
    best_epoch: int = int(np.argmin(val_losses))

    # Save the training and validation losses plot
    plot_losses(train_losses, val_losses, folder=save_folder, filename='losses_epoch_150.png')

    # Save the training and validation losses to a JSON file
    save_losses(train_losses, val_losses, folder=save_folder, filename='losses_epoch_150.json')

    # Load the best model and evaluate it on the test set
    model_path: str = os.path.join(save_folder, 'best_model.pth')
    model.load_state_dict(torch.load(model_path))
    test_loss: float = evaluate_model(model, test_loader)
    relative_test_error: float = calculate_relative_test_error(model, test_loader)
    relative_test_error_squared: float = calculate_relative_test_error(model, test_loader, squared=True)

    # Print evaluation results
    print(f'Test Loss: {test_loss}')
    print(f'Relative Test Error: {relative_test_error}')
    print(f'Relative Test Error (Squared): {relative_test_error_squared}')
    print(f'Training Time: {training_time} seconds')
    print(f'Number of Parameters: {num_params}')

    # Save additional metrics to a JSON file
    results: Dict[str, Any] = {
        "training_time": training_time,
        "num_epochs": num_epochs,
        "best_epoch": best_epoch,
        "relative_test_error": relative_test_error,
        "relative_test_error_squared": relative_test_error_squared,
        "num_params": num_params
    }

    with open(os.path.join(save_folder, 'additional_metrics.json'), 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
