import torch.nn as nn
import torch
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Construct a MLP layer, including 
    1. a single fully-connected layer,
    2. layer normalization 
    3. ReLU

    This MLP is specifically used for the PolylineSubGraphLayer for VectorNet.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 64
        ) -> None:
        """
        Args:
            input_size: the size of input layer.
            output_size: the size of output layer.
            hidden_size (Optional): the size of output layer.
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x:torch.Tensor):
        """
        Args:
            x: input
            [batch_size, num_vectors, input_size]
        Returns:
            x (torch.Tensor): output
            [batch_size, num_polylines, num_vectors, output_size]
        """
        # [batch_size, .., input_size] -> [batch_size, .., hidden_size]
        x = self.fc1(x)
        # [batch_size, .., hidden_size]
        x = self.norm(x)
        # [batch_size, .., hidden_size]
        x = F.relu(x)
        # [batch_size, .., output_size]
        x = self.fc2(x)
        return x
