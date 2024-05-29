import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class FourierLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Initialize the FourierLayer with specified input and output channels.
        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        """
        super(FourierLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FourierLayer.
        :param x: Input tensor.
        :return: Transformed tensor after applying FFT and inverse FFT.
        """
        # Perform FFT
        x_fft = torch.fft.fft(x)

        # Apply learned weights in Fourier space
        x_fft_weighted = x_fft.unsqueeze(-1) * self.weight

        # Perform inverse FFT
        x_ifft = torch.fft.ifft(x_fft_weighted.sum(dim=-2))

        return x_ifft.real


class HybridBranchNet(nn.Module):
    def __init__(self, branch_layers: List[int], fourier_features: int) -> None:
        """
        Initialize the HybridBranchNet with specified branch layers and Fourier features.
        :param branch_layers: List of integers defining the architecture of the branch network.
        :param fourier_features: Number of features after the Fourier transformation.
        """
        super(HybridBranchNet, self).__init__()

        # Initial fully connected layers
        self.initial_fc_layers = self._build_network(branch_layers[:-1])

        # Fourier layer applied after the initial fully connected layers
        self.fourier_layer = FourierLayer(branch_layers[-2], fourier_features)

        # Additional fully connected layers after Fourier transformation, if needed
        self.final_fc_layers = nn.Linear(fourier_features, branch_layers[-1])

    def _build_network(self, layers: List[int]) -> nn.Sequential:
        """
        Helper function to create a fully connected network from a list of layers.
        :param layers: List of integers specifying the number of units in each layer.
        :return: Sequential model with fully connected layers.
        """
        network_layers = []
        for i in range(len(layers) - 1):
            network_layers.append(nn.Linear(layers[i], layers[i + 1]))
            network_layers.append(nn.ReLU())
        return nn.Sequential(*network_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the HybridBranchNet.
        :param x: Input tensor.
        :return: Output tensor after passing through initial, Fourier, and final layers.
        """
        x = self.initial_fc_layers(x)
        x = self.fourier_layer(x)
        x = self.final_fc_layers(x)
        return x


class DeepONets(nn.Module):
    def __init__(self, branch_layers: List[int], trunk_layers: List[int], out_features: int) -> None:
        """
        Initializes the DeepONets model.
        :param branch_layers: List of integers defining the architecture of the branch network.
        :param trunk_layers: List of integers defining the architecture of the trunk network.
        :param out_features: Number of output features (solution dimension).
        """
        super(DeepONets, self).__init__()

        # Initialize the branch net
        self.branch_net = self._build_network(branch_layers)

        # Initialize the trunk net
        self.trunk_net = self._build_network(trunk_layers)

        # The final linear layer that combines branch and trunk net outputs
        self.out_layer = nn.Linear(branch_layers[-1], out_features)

    def _build_network(self, layers: List[int]) -> nn.Sequential:
        """
        Helper function to create a fully connected network from a list of layers.
        :param layers: List of integers specifying the number of units in each layer.
        :return: Sequential model with fully connected layers.
        """
        network_layers = []
        for i in range(len(layers) - 1):
            network_layers.append(nn.Linear(layers[i], layers[i + 1]))
            network_layers.append(nn.LeakyReLU(0.1))
        return nn.Sequential(*network_layers)

    def forward(self, parameter: torch.Tensor, coordinates_time: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DeepONets.
        :param parameter: Tensor containing the parameter (shape: [batch_size, parameter_dim]).
        :param coordinates_time: Tensor containing the coordinates and time (shape: [batch_size, coordinates_time_dim]).
        :return: Predicted solution at the given coordinates and time for the given parameter.
        """
        # Pass the inputs through the branch and trunk networks
        branch_output = self.branch_net(parameter)
        trunk_output = self.trunk_net(coordinates_time)
        branch_output = branch_output.unsqueeze(1).expand(-1, trunk_output.size(1), -1)

        # Pass the combined input through the final linear layer
        output = self.out_layer(branch_output * trunk_output).squeeze(-1)
        return output

    def extract_features(self, parameter: torch.Tensor, coordinates_time: torch.Tensor) -> torch.Tensor:
        """
        Extract combined features from the branch and trunk networks.
        :param parameter: Tensor containing the parameter.
        :param coordinates_time: Tensor containing the coordinates and time.
        :return: Combined features tensor.
        """
        with torch.no_grad():
            trunk_output = self.trunk_net(coordinates_time)
            branch_output = self.branch_net(parameter)
            branch_output = branch_output.unsqueeze(1).expand(-1, trunk_output.size(1), -1)

            combined_output = branch_output * trunk_output
        return combined_output


class DeepONetsWithFNO(nn.Module):
    def __init__(self, branch_layers: List[int], trunk_layers: List[int], out_features: int) -> None:
        """
        Initializes the DeepONets model with a Fourier Layer as the branch network.
        :param branch_layers: List of integers defining the architecture of the branch network.
        :param trunk_layers: List of integers defining the architecture of the trunk network.
        :param out_features: Number of output features (solution dimension).
        """
        super(DeepONetsWithFNO, self).__init__()

        # Initialize the branch net with FNO
        self.branch_net = FourierLayer(branch_layers[0], branch_layers[-1])

        # Initialize the trunk net
        self.trunk_net = self._build_network(trunk_layers)

        # The final linear layer that combines branch and trunk net outputs
        self.out_layer = nn.Linear(trunk_layers[-1], out_features)

    def _build_network(self, layers: List[int]) -> nn.Sequential:
        """
        Helper function to create a fully connected network from a list of layers.
        :param layers: List of integers specifying the number of units in each layer.
        :return: Sequential model with fully connected layers.
        """
        network_layers = []
        for i in range(len(layers) - 1):
            network_layers.append(nn.Linear(layers[i], layers[i + 1]))
            network_layers.append(nn.LeakyReLU(0.1))
        return nn.Sequential(*network_layers)

    def forward(self, parameter: torch.Tensor, coordinates_time: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DeepONets with FNO as the branch network.
        :param parameter: Tensor containing the parameter.
        :param coordinates_time: Tensor containing the coordinates and time.
        :return: Predicted solution tensor.
        """
        # Pass the parameter through the branch network (FNO)
        branch_output = self.branch_net(parameter)

        # Pass the coordinates_time through the trunk network
        trunk_output = self.trunk_net(coordinates_time)

        # Ensure dimensions match for multiplication
        branch_output = branch_output.unsqueeze(1).expand(-1, trunk_output.size(1), -1)

        # Combine branch and trunk outputs
        combined_output = branch_output * trunk_output

        # Pass the combined output through the final layer
        output = self.out_layer(combined_output).squeeze(-1)

        return output


class DeepONetsWithHybridBranch(nn.Module):
    def __init__(self, branch_layers: List[int], trunk_layers: List[int], out_features: int, fourier_features: int) -> None:
        """
        Initializes the DeepONets model with a hybrid branch network.
        :param branch_layers: List of integers defining the architecture of the branch network.
        :param trunk_layers: List of integers defining the architecture of the trunk network.
        :param out_features: Number of output features (solution dimension).
        :param fourier_features: Number of features after the Fourier transformation in the hybrid branch.
        """
        super(DeepONetsWithHybridBranch, self).__init__()

        # Initialize the hybrid branch net
        self.branch_net = HybridBranchNet(branch_layers, fourier_features)

        # Initialize the trunk net
        self.trunk_net = self._build_network(trunk_layers)

        # The final linear layer that combines branch and trunk net outputs
        self.out_layer = nn.Linear(trunk_layers[-1], out_features)

    def _build_network(self, layers: List[int]) -> nn.Sequential:
        """
        Helper function to create a fully connected network from a list of layers.
        :param layers: List of integers specifying the number of units in each layer.
        :return: Sequential model with fully connected layers.
        """
        network_layers = []
        for i in range(len(layers) - 1):
            network_layers.append(nn.Linear(layers[i], layers[i + 1]))
            network_layers.append(nn.ReLU())
        return nn.Sequential(*network_layers)

    def forward(self, parameter: torch.Tensor, coordinates_time: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DeepONets with a hybrid branch network.
        :param parameter: Tensor containing the parameter.
        :param coordinates_time: Tensor containing the coordinates and time.
        :return: Predicted solution tensor.
        """
        # Pass the parameter through the hybrid branch network
        branch_output = self.branch_net(parameter)

        # Pass the coordinates_time through the trunk network
        trunk_output = self.trunk_net(coordinates_time)

        # Ensure dimensions match for multiplication
        branch_output = branch_output.unsqueeze(1).expand(-1, trunk_output.size(1), -1)

        # Combine branch and trunk outputs
        combined_output = branch_output * trunk_output

        # Pass the combined output through the final layer
        output = self.out_layer(combined_output).squeeze(-1)

        return output


# Example of defining a DeepONets model
# branch_layers = [1, 20, 20, 10]  # 1 input feature (parameter), 2 hidden layers with 20 units each, 10 units in the output layer
# trunk_layers = [3, 50, 50, 10]   # 3 input features (2 coordinates + time), 2 hidden layers with 50 units each, 10 units in the output layer
# out_features = 1  # Predicting a single value (solution) at each point

# model = DeepONets(branch_layers, trunk_layers, out_features)
