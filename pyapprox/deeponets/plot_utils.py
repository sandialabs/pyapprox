import os
import json
import matplotlib.pyplot as plt
from typing import List


def plot_losses(train_losses: List[float], val_losses: List[float], folder: str = 'plots', filename: str = 'training_validation_losses.png') -> None:
    """
    Plot and save the training and validation losses.

    Parameters:
    train_losses (List[float]): List of training losses.
    val_losses (List[float]): List of validation losses.
    folder (str): Directory to save the plot. Default is 'plots'.
    filename (str): Filename for the saved plot. Default is 'training_validation_losses.png'.
    """
    # Check if the folder exists, if not, create it
    if not os.path.exists(folder):
        os.makedirs(folder)

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')  # Set the y-axis to logarithmic scale
    plt.legend()
    plt.grid(True)

    # Save the plot instead of showing it
    plt.savefig(os.path.join(folder, filename))
    plt.close()


def load_validation_losses(folder: str, filename: str = 'losses_epoch_150.json') -> List[float]:
    """
    Load validation losses from a JSON file.

    Parameters:
    folder (str): Directory containing the JSON file.
    filename (str): Name of the JSON file. Default is 'losses_epoch_150.json'.

    Returns:
    List[float]: List of validation losses.
    """
    filepath = os.path.join(folder, filename)
    with open(filepath, 'r') as file:
        data = json.load(file)
    # Extract validation losses
    val_losses = [entry['val_loss'] for entry in data]
    return val_losses


def plot_multiple_validation_losses(folders: List[str], folder: str = 'plots', filename: str = 'comparison_validation_losses.png') -> None:
    """
    Plot and save the comparison of validation losses across multiple models.

    Parameters:
    folders (List[str]): List of directories containing validation loss JSON files.
    folder (str): Directory to save the plot. Default is 'plots'.
    filename (str): Filename for the saved plot. Default is 'comparison_validation_losses.png'.
    """
    # Check if the folder exists, if not, create it
    if not os.path.exists(folder):
        os.makedirs(folder)

    plt.figure(figsize=(10, 6))

    for i, folder_path in enumerate(folders):
        # Load validation losses from each specified folder
        val_losses = load_validation_losses(folder_path)
        # Use the last part of the folder path as the label
        label = os.path.basename(folder_path)
        plt.plot(val_losses, label=f'Validation Loss - {label}')

    plt.title('Comparison of Validation Losses Across Models')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')  # Set the y-axis to logarithmic scale
    plt.legend()
    plt.grid(True)

    # Save the plot instead of showing it
    plt.savefig(os.path.join(folder, filename))
    plt.close()
