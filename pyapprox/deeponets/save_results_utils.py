import json
import os
from typing import List

def save_losses(train_losses: List[float], val_losses: List[float], folder: str, filename: str) -> None:
    """
    Saves training and validation losses to a JSON file.

    Parameters:
    train_losses (List[float]): List of training losses.
    val_losses (List[float]): List of validation losses.
    folder (str): Directory to save the JSON file.
    filename (str): Name of the JSON file.
    """
    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)
    
    # Define the path for the JSON file
    filepath = os.path.join(folder, filename)
    
    # Prepare data as a list of dictionaries
    data = [{'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss}
            for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses))]
    
    # Write losses to the JSON file
    with open(filepath, 'w') as file:
        json.dump(data, file, indent=4)
    
    print(f"Losses saved to {filepath}")
