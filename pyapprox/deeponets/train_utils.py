import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Tuple, List, Optional


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a PyTorch model.

    Parameters:
    model (torch.nn.Module): The PyTorch model.

    Returns:
    int: The number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_relative_test_error(model: nn.Module, data_loader: torch.utils.data.DataLoader, squared: bool = False) -> float:
    """
    Calculate the relative test error of a model on a dataset.

    Parameters:
    model (torch.nn.Module): The PyTorch model.
    data_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
    squared (bool): Whether to calculate squared error or absolute error. Default is False.

    Returns:
    float: The relative test error.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    total_error = 0.0
    total_samples = 0

    with torch.no_grad():
        for parameters, coordinates_time, solutions in data_loader:
            parameters = parameters.to(device)
            coordinates_time = coordinates_time.to(device)
            solutions = solutions.to(device)

            outputs = model(parameters, coordinates_time)
            if squared:
                error = torch.sum((outputs - solutions.squeeze(-1)) ** 2).item()
            else:
                error = torch.sum(torch.abs(outputs - solutions.squeeze(-1))).item()
            total_error += error
            total_samples += solutions.size(0)

    relative_error = total_error / total_samples
    return relative_error


def evaluate_model(model: nn.Module, data_loader: torch.utils.data.DataLoader) -> float:
    """
    Evaluate the model on a dataset and calculate the mean squared error.

    Parameters:
    model (torch.nn.Module): The PyTorch model.
    data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.

    Returns:
    float: The mean squared error of the model on the dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0

    with torch.no_grad():
        for parameters, coordinates_time, solutions in data_loader:
            parameters = parameters.to(device)
            coordinates_time = coordinates_time.to(device)
            solutions = solutions.to(device)

            outputs = model(parameters, coordinates_time)
            loss = criterion(outputs, solutions.squeeze(-1))

            total_loss += loss.item() * parameters.size(0)

    return total_loss / len(data_loader.dataset)


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    validation_loader: Optional[torch.utils.data.DataLoader],
    save_folder: str,
    epochs: int = 10,
    learning_rate: float = 0.001
) -> Tuple[List[float], List[float]]:
    """
    Train the model and save the best model based on validation loss.

    Parameters:
    model (torch.nn.Module): The PyTorch model.
    train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
    validation_loader (Optional[torch.utils.data.DataLoader]): DataLoader for the validation dataset.
    save_folder (str): Directory to save the best model.
    epochs (int): Number of training epochs. Default is 10.
    learning_rate (float): Learning rate for the optimizer. Default is 0.001.

    Returns:
    Tuple[List[float], List[float]]: Lists of training and validation losses.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.00005)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    os.makedirs(save_folder, exist_ok=True)
    model_path = os.path.join(save_folder, 'best_model.pth')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for parameters, coordinates_time, solutions in train_loader:
            parameters = parameters.to(device)
            coordinates_time = coordinates_time.to(device)
            solutions = solutions.to(device)

            optimizer.zero_grad()
            outputs = model(parameters, coordinates_time)
            loss = criterion(outputs, solutions.squeeze(-1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * parameters.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        if validation_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for parameters, coordinates_time, solutions in validation_loader:
                    parameters = parameters.to(device)
                    coordinates_time = coordinates_time.to(device)
                    solutions = solutions.to(device)

                    outputs = model(parameters, coordinates_time)
                    loss = criterion(outputs, solutions.squeeze(-1))

                    val_loss += loss.item() * parameters.size(0)

            epoch_val_loss = val_loss / len(validation_loader.dataset)
            val_losses.append(epoch_val_loss)

            # Save the best model
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                torch.save(model.state_dict(), model_path)

        print(f'Epoch {epoch + 1}, Loss: {epoch_loss}, Validation Loss: {epoch_val_loss if validation_loader is not None else "N/A"}')

        scheduler.step()

    print('Finished Training')
    return train_losses, val_losses
