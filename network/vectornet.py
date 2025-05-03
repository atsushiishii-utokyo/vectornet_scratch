import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorNet(nn.Module):
    """
    VectorNet Class for the agents trajectories prediction
    """
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        output_dim: int = 2
        ) -> None:
        """
        Instatiate the VectorNet
        Construct Polyline Subgraph (Local graph),
        Global Graph, and Decoder.

        Args:
            input_dim (int): input dimmension of the local graph
            hidden_dim (int): Hidden dimension for the network layers
            output_dim (int): Output dimension of the VectorNet network
        """
        super().__init__()
        self.polyline_graph = PolylineSubGraph(input_dim, hidden_dim)
        self.global_graph = GlobalGraph(hidden_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        
    def forward(
        self,
        polylines: torch.Tensor,
        polyline_masks: torch.Tensor
        ) -> torch.Tensor:
        # Process each polyline through the subgraph
        polyline_features = self.polyline_graph(polylines, polyline_masks)
        
        # Process through global graph
        global_features = self.global_graph(polyline_features)
        
        # Decode to output
        output = self.decoder(global_features)
        return output

class PolylineSubGraph(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.node_encoder = nn.Linear(input_dim, hidden_dim)
        self.edge_encoder = nn.Linear(input_dim * 2, hidden_dim)
        self.node_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, polylines: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        # polylines: [batch_size, num_polylines, num_nodes, input_dim]
        # masks: [batch_size, num_polylines, num_nodes]
        
        batch_size, num_polylines, num_nodes, _ = polylines.shape
        
        # Encode nodes
        node_features = self.node_encoder(polylines)
        
        # Create edge features
        edge_features = torch.cat([
            polylines.unsqueeze(3).expand(-1, -1, -1, num_nodes, -1),
            polylines.unsqueeze(2).expand(-1, -1, num_nodes, -1, -1)
        ], dim=-1)
        
        # Process through node processor
        node_features = self.node_processor(node_features)
        
        # Apply masks
        node_features = node_features * masks.unsqueeze(-1)
        
        # Aggregate node features to get polyline features
        polyline_features = node_features.sum(dim=2) / (masks.sum(dim=2, keepdim=True) + 1e-6)
        
        return polyline_features

class GlobalGraph(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads=8)
        self.processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, polyline_features: torch.Tensor) -> torch.Tensor:
        # polyline_features: [batch_size, num_polylines, hidden_dim]
        
        # Reshape for attention
        batch_size, num_polylines, hidden_dim = polyline_features.shape
        polyline_features = polyline_features.reshape(batch_size * num_polylines, 1, hidden_dim)
        
        # Apply attention
        attended_features, _ = self.attention(polyline_features, polyline_features, polyline_features)
        
        # Process through MLP
        attended_features = attended_features.reshape(batch_size, num_polylines, hidden_dim)
        global_features = self.processor(attended_features)
        
        return global_features

