import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int) -> None:
        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param modes1: Number of Fourier modes to multiply, at most floor(N/2) + 1.
        """
        super(SpectralConv1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def compl_mul1d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Complex multiplication of input with weights.
        :param input: Input tensor.
        :param weights: Weights tensor.
        :return: Result of the complex multiplication.
        """
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SpectralConv1d layer.
        :param x: Input tensor.
        :return: Output tensor after Fourier transformation and inverse Fourier transformation.
        """
        batchsize = x.shape[0]

        # Compute Fourier coefficients up to a factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNOBranchNet(nn.Module):
    def __init__(self, branch_layers: List[int], fourier_features: int, modes: int) -> None:
        """
        A network structure inspired by FNO1d, focusing on learning mappings from spatial/temporal inputs to outputs.
        Includes multiple spectral convolution layers followed by linear transformations.
        :param branch_layers: List of integers defining the architecture of the branch network.
        :param fourier_features: Number of features after the Fourier transformation.
        :param modes: Number of Fourier modes.
        """
        super(FNOBranchNet, self).__init__()

        self.fc0 = nn.Linear(1, fourier_features)  # Lift the input to the desired channel dimension

        # Four layers of SpectralConv1d, similar to the four layers of integral operators in FNO1d
        self.conv_layers = nn.ModuleList([
            SpectralConv1d(fourier_features, fourier_features, modes) for _ in range(2)
        ])

        # Corresponding pointwise convolution layers
        self.w_layers = nn.ModuleList([
            nn.Conv1d(fourier_features, fourier_features, 1) for _ in range(2)
        ])

        # Projection from the channel space to the output space
        self.fc1 = nn.Linear(fourier_features, 32)
        self.fc2 = nn.Linear(32, branch_layers[-1])  # Adjust the final output dimension as needed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FNOBranchNet.
        :param x: Input tensor.
        :return: Transformed output tensor.
        """
        if x.ndim == 2:
            x = x.unsqueeze(2)  # Add a dummy dimension for Conv1d if needed

        x = self.fc0(x)
        x = x.permute(0, 2, 1)  # Adjust dimension for conv1d

        # Process through spectral and pointwise convolution layers with activations
        for conv_layer, w_layer in zip(self.conv_layers, self.w_layers):
            x1 = conv_layer(x)
            x2 = w_layer(x)
            x = x1 + x2
            x = F.gelu(x)  # Activation

        x = x.permute(0, 2, 1)  # Re-adjust dimension to match linear layers
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class DeepONetsWithFNOBranch(nn.Module):
    def __init__(self, branch_layers: List[int], trunk_layers: List[int], out_features: int, fourier_features: int, modes: int) -> None:
        """
        Initialize the DeepONets model with an FNO branch network.
        :param branch_layers: List of integers defining the architecture of the branch network.
        :param trunk_layers: List of integers defining the architecture of the trunk network.
        :param out_features: Number of output features (solution dimension).
        :param fourier_features: Number of features after the Fourier transformation in the branch network.
        :param modes: Number of Fourier modes.
        """
        super(DeepONetsWithFNOBranch, self).__init__()

        # Initialize the FNO branch net
        self.branch_net = FNOBranchNet(branch_layers, fourier_features, modes)

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
        Forward pass of the DeepONets with an FNO branch network.
        :param parameter: Tensor containing the parameter.
        :param coordinates_time: Tensor containing the coordinates and time.
        :return: Predicted solution tensor.
        """
        # Pass the parameter through the FNO branch network
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
