import torch
import torch.nn as nn
import torch.nn.functional as F
from mlp import MLP
from config import VECTORNET_CONFIG as config
from typing import Optional

class VectorNet_prediction(nn.Module):
    """
    VectorNet Class for the agents trajectories prediction packaging the VectorNet and the MLP.
    
    MLP is used to decode the output of the VectorNet to the prediction output.
    [batch_size, num_features * (2**num_subgraph_layers)] -> 
    [batch_size, num_prediction_features * num_future_steps]

    Args:
        config: Configuration for the VectorNet.
    """
    def __init__(self, config: dict[str, int]) -> None:
        super().__init__()
        self.vectornet = VectorNet(
            num_vectors=config["num_vectors"],
            num_features=config["num_features"],
            num_subgraph_layers=config["num_subgraph_layers"],
            num_global_layers=config["num_global_layers"],
            num_global_heads=config["num_global_heads"],
        )
        self.traj_decoder = MLP(
            input_size=self.vectornet.graph_output_dim,
            output_size=config["num_prediction_features"]*config["num_future_steps"]
        )
    def forward(self, 
        polylines_list: list[torch.Tensor],
        target_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward the inputs of VectorNet and output the prediction.

        Args:
            polylines_list: List of polylines in the history as inputs.
                [[num_vectors (num_past_steps), num_features] * num_polylines]
                num_polylines is the number of agents + Map related polylines in the scene.
                Inputs should be encoded to the vectors in advance to be used in the Graph Neural Network.

            target_index: Index of the target agent to be predicted, mapping to the batch index.
                [batch_size]
                (ex) target_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,...15] if batch_size = 16. 
                Since VectorNet could only predict trajectory of one agent at a time,
                we need to specify the index of the target agent to be predicted.
                This index is used to select the correct output from the VectorNet.
                Batch processing allows us to predict trajectories of multiple agents at once.
                However, the inputs of Polylines should be converted to the target agent centric coordinates.
                This is done by subtracting the position of the target agent from the position of the other agents.
                This is done by the following code:
                polylines_list = [polylines - polylines[target_index]]

                TODO: Implement the code to convert the polylines to the target agent centric coordinates.

        Returns:
            output (torch.Tensor): Trajectory Prediction
        """
        # [[num_vectors, num_features] * num_polylines] -> [batch_size, graph_output_dim]
        output = self.vectornet(polylines_list)
        # [batch_size, graph_output_dim] -> [batch_size, num_prediction_features * num_future_steps]
        output = self.traj_decoder(output)
        return output

class VectorNet(nn.Module):
    """
    VectorNet Class for the agents trajectories prediction
    """
    def __init__(
        self,
        num_vectors: int,
        num_features: int,
        num_subgraph_layers: int,
        num_global_layers: int,
        num_global_heads: int,
    ) -> None:
        """
        Instantiate the VectorNet
        Construct Polyline Subgraph (Local graph),
        Global Graph, and Decoder.

        Args:
            num_vectors: number of vectors in the local graph.
                corresponds to the number of past steps to input to the model.
                Same as num_past_steps in the config.
            num_features: The length of vector v (ds_x,ds_y,de_x,de_y,a,j).
                corresponds to the size of the last dimension of the local graph.
            num_subgraph_layers: number of layers for the polyline subgraph.
            num_global_layers: number of layers for the global graph.
            num_global_heads: number of heads for the global graph.
        """
        super().__init__()
        self.polyline_graph = PolylineSubGraph(
            num_subgraph_layers=num_subgraph_layers,
            num_features=num_features
        )
        self.global_graph = GlobalGraph(hidden_dim, hidden_dim)
        # Compute the output dimension of Global graph 
        # using the number of features and the number of layers in the PolylneSubGraph
        self.graph_output_dim = num_features * (2**num_subgraph_layers)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        
    def forward(
        self,
        polylines_list: list[torch.Tensor],
        polyline_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward the inputs of VectorNet and output the prediction.
        Except for the prediction inference, mask out the node arbitarily for the graph completion task.
        # [[num_vectors, num_features] * num_polylines] -> [batch_size, graph_output_dim]

        Args:
            polylines_list: Subset of polylinses; P = [P1, P2,...,Pn]}
                Each P (Polyline) is expressed by subset of vectors Pi = {v0, v1, ...vp}
                where the shape of Pi is [num_vectors, num_features]

            polyline_masks: Mask for the polylines for the graph completion task.
        Returns:
            output (torch.Tensor): Trajectory Prediction
        """
        # Process each polyline through the subgraph
        polyline_features = self.polyline_graph(polylines_list)
        
        # Process through global graph
        global_features = self.global_graph(polyline_features)
        
        # Decode to output
        output = self.decoder(global_features)
        return output

class PolylineSubGraph(nn.Module):
    def __init__(self, num_subgraph_layers: int, num_features: int) -> None:
        """
        Instantiate the PolylineSubGraph.
        Here, the local features of each polyline is computed.
        This comprises of multiple layers of PolylineSubGraphLayer.
 
        Args:
            num_subgraph_layers: number of layers for the polyline subgraph.
            num_features: the dimension of the vector v (ds_x,ds_y,de_x,de_y,a,j).
                corresponds to the size of the last dimension of the local graph.
        """
        super().__init__()
        self.num_subgraph_layers = num_subgraph_layers
        self.num_features = num_features
        # Create a list of PolylineSubGraphLayer for each layer
        # The input dimension of the i-th layer is num_features * (2**i)
        # In each layer, it computes
        # [batch_size, num_polylines, num_vectors, num_features * (2**i)] -> 
        # [batch_size, num_polylines, num_vectors, num_features * (2**(i+1))]
        self.polyline_subgraph_layers = nn.ModuleList([
            PolylineSubGraphLayer(num_features*(2**i)) for i in range(num_subgraph_layers)
        ])
        
    def forward(self, polylines_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Forward the inputs of PolylineSubGraph and output the features of each polyline.

        Args:
            polylines_list: List of polylines in the history as inputs.
                [[num_vectors (num_past_steps), num_features] * num_polylines]
                num_polylines is the number of agents + Map related polylines in the scene.
                Inputs should be encoded to the vectors in advance to be used in the Graph Neural Network.

        Returns:
            polyline_features_list: Features of each polyline.
                [[batch_size, num_features * (2**num_subgraph_layers)]*num_polylines]
        """
        batch_size = polylines_list[0].shape[0]
        num_polylines = len(polylines_list)
        
        # 1. Encode nodes
        node_features_list = [self.node_encoder(polylines) for polylines in polylines_list]
        
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

class PolylineSubGraphLayer(nn.Module):
    """
    A single layer of the PolylineSubGraph in VectorNet.
    
    This layer processes each polyline through node encoding, edge encoding,
    and node processing steps to extract local features. It includes:
    1. Node encoding using MLP
    2. Edge feature computation between nodes
    3. Node feature processing
    4. Feature aggregation across nodes
    
    The layer takes polyline vectors as input and outputs processed features
    that capture the local structure and relationships within each polyline.
    """
    def __init__(self,
        input_dim: int,
    ) -> None:
        """
        Args:
            input_dim: input dimension of the local graph.
        """
        super().__init__()
        self.node_encoder = MLP(input_dim, input_dim)
        self.edge_encoder = nn.Linear(input_dim * 2, hidden_dim)
        self.node_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x: torch.Tensor):
        """
        x (torch.Tensor): A set of vectors, input of SubGraph layer
            [batch_size, n, input_dim]
        Returns:
            x (torch.Tensor): Processed features of each vector
                [batch_size, n, hidden_dim]
        """
        x = self.node_encoder(x)
        # Concat wit the max pooling

        return x

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
        """Process polyline features through global attention and MLP layers.
        
        Args:
            polyline_features (torch.Tensor): Input tensor containing polyline features
                Shape: [batch_size, num_polylines, hidden_dim]
                
        Returns:
            torch.Tensor: Processed global features after attention and MLP
                Shape: [batch_size, num_polylines, hidden_dim]
        """
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

