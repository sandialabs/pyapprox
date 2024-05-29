import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint


class ODEFunc(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int) -> None:
        """
        Initialize the ODE function network.
        :param in_features: Number of input features.
        :param hidden_dim: Dimension of hidden layers.
        """
        super(ODEFunc, self).__init__()
        # Expanded network with more layers
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2 * hidden_dim),  # Additional layer
            nn.Tanh(),
            nn.Linear(2 * hidden_dim, hidden_dim),  # Additional layer
            nn.Tanh(),
            nn.Linear(hidden_dim, in_features),
        )
    
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ODE function.
        :param t: Time tensor.
        :param x: Input tensor.
        :return: Output tensor after passing through the network.
        """
        return self.net(x)


class ODEBranchNet(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int) -> None:
        """
        Initialize the ODE branch network.
        :param in_features: Number of input features.
        :param hidden_dim: Dimension of hidden layers.
        """
        super(ODEBranchNet, self).__init__()
        self.odefunc = ODEFunc(in_features, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ODE branch network.
        :param x: Input tensor.
        :return: Output tensor after integrating the ODE.
        """
        # Integrate from t=0 to t=1
        t = torch.linspace(0., 1., 50)
        out = odeint(self.odefunc, x, t, method='dopri5')
        return out[-1]


class DeepONetsWithODEBranch(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, trunk_layers: list[int], out_features: int) -> None:
        """
        Initialize the DeepONets model with an ODE branch network.
        :param in_features: Number of input features for the branch network.
        :param hidden_dim: Dimension of hidden layers in the branch network.
        :param trunk_layers: List of integers defining the architecture of the trunk network.
        :param out_features: Number of output features (solution dimension).
        """
        super(DeepONetsWithODEBranch, self).__init__()
        
        # Initialize the ODE branch net
        self.branch_net = ODEBranchNet(in_features, hidden_dim)
        
        # Initialize the trunk net
        self.trunk_net = self._build_network(trunk_layers)
        
        # The final linear layer that combines branch and trunk net outputs
        self.out_layer = nn.Linear(trunk_layers[-1], out_features)

    def _build_network(self, layers: list[int]) -> nn.Sequential:
        """
        Helper function to create a fully connected network from a list of layers.
        :param layers: List of integers specifying the number of units in each layer.
        :return: Sequential model with fully connected layers.
        """
        network_layers = []
        for i in range(len(layers) - 1):
            network_layers.append(nn.Linear(layers[i], layers[i+1]))
            network_layers.append(nn.ReLU())
        return nn.Sequential(*network_layers)

    def forward(self, parameter: torch.Tensor, coordinates_time: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DeepONets with an ODE branch network.
        :param parameter: Tensor containing the parameter.
        :param coordinates_time: Tensor containing the coordinates and time.
        :return: Predicted solution tensor.
        """
        # Pass the parameter through the ODE branch network
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
