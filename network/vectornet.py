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
    [batch_size, num_future_steps, num_prediction_features]

    Args:
        config: Configuration for the VectorNet.
    """
    def __init__(self, config: dict[str, int]) -> None:
        super().__init__()
        self.num_future_steps = config["num_future_steps"]
        self.num_prediction_features = config["num_prediction_features"]
        self.vectornet = VectorNet(
            num_vectors=config["num_vectors"],
            num_features=config["num_features"],
            num_subgraph_layers=config["num_subgraph_layers"],
            num_global_layers=config["num_global_layers"],
            num_global_heads=config["num_global_heads"],
        )
        self.traj_decoder = MLP(
            input_size=self.vectornet.global_graph_input_output_dim,
            output_size=self.num_future_steps*self.num_prediction_features
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
                with the following code:
                polylines_list = [polylines - polylines[target_index]]

                TODO: Implement the code to convert the polylines to the target agent centric coordinates.

        Returns:
            output: Trajectory Prediction for each given track id
                [batch_size, num_future_steps, num_prediction_features]
        """
        # [[num_vectors, num_features] * num_polylines] -> [batch_size, graph_output_dim]
        output = self.vectornet(polylines_list, target_index)
        # [batch_size, graph_output_dim] -> [batch_size, num_prediction_features * num_future_steps]
        output = self.traj_decoder(output)
        # reshape the output
        output = output.view(-1, self.num_future_steps, self.num_prediction_features)

        return output

class VectorNet(nn.Module):
    """
    VectorNet Class for the agents trajectories prediction.
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
        Instantiate the VectorNet.

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
            num_global_heads: number of attention heads for the global graph.
        """
        super().__init__()
        # Compute the output dimension of Global graph 
        # using the number of features and the number of layers in the PolylneSubGraph
        self.global_graph_input_output_dim = num_features * (2**num_subgraph_layers)
        self.polyline_sub_graph = PolylineSubGraph(
            num_subgraph_layers=num_subgraph_layers,
            num_features=num_features
        )
        self.global_graph = GlobalGraph(
            num_global_layers=num_global_layers,
            num_global_heads=num_global_heads,
            emb_dim=self.global_graph_input_output_dim,
        )
        
    def forward(
        self,
        polylines_list: list[torch.Tensor],
        target_index: torch.Tensor,
        polyline_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward the inputs of VectorNet and output the prediction.

        (1) Process each polyline through the subgraph.
        This returns the features of each polyline with the shape of 
        [batch_size, num_features * (2**num_subgraph_layers)].
        
        (2) Stack the features of each polyline to form a global graph with the shape of 
        [batch_size, num_polylines, num_features * (2**num_subgraph_layers)].

        (3) Process through global graph.
        This returns the features of each target agent with the shape of
        [batch_size, global_graph_input_output_dim].

        Except for the prediction inference, mask out the node arbitarily for the graph completion task.

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

            polyline_masks (Optional): Mask for the polylines for the graph completion task.
        Returns:
            output: Global features of each target agent
                [batch_size, global_graph_input_output_dim]
                Note that global_graph_input_output_dim = num_features * (2**num_subgraph_layers),
                same as the output dimension of the PolylineSubGraph and input dimension of the GlobalGraph.
        """
        # (1) Process each polyline through the subgraph
        self.batch_size = target_index.shape[0]
        polyline_features_list = []
        # Process each polyline through the subgraph
        for polyline in polylines_list:
            # [num_vectors, num_features] -> [batch_size, num_vectors, num_features]
            polyline = polyline.unsqueeze(0).repeat(self.batch_size, 1, 1)
            # [batch_size, num_vectors, num_features] -> [batch_size, num_vectors, num_features]
            # Convert to the target agent centric coordinates
            polyline = polyline - polyline[target_index]
            polyline_features = self.polyline_sub_graph(polyline)
            polyline_features_list.append(polyline_features)
        
        # (2) Stack the features of each polyline
        # [batch_size, num_features * (2**num_subgraph_layers)] * num_polylines
        polyline_sub_graph_features = torch.stack(polyline_features_list, dim=1)
        assert polyline_sub_graph_features.shape == (self.batch_size, len(polylines_list), self.global_graph_input_output_dim)

        # (3) Process through global graph
        global_features = self.global_graph(
            polyline_features=polyline_sub_graph_features,
            target_index=target_index,
        )
        
        return global_features

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
        #   [batch_size, num_vectors, num_features * (2**i)] -> 
        #   [batch_size, num_vectors, num_features * (2**(i+1))]
        self.polyline_subgraph_layers = nn.ModuleList([
            PolylineSubGraphLayer(num_features*(2**i)) for i in range(num_subgraph_layers)
        ])
        
    def forward(self, polyline: torch.Tensor) -> torch.Tensor:
        """
        Forward the inputs of PolylineSubGraph and output the features of each polyline.
        First, process each polyline through the subgraph layers.
        This returns the features of each vector in the polyline 
        with the shape of [batch_size, num_vectors, num_features * (2**num_subgraph_layers)].

        Then, aggregate the features of each polyline using max pooling.
        This returns the features of each polyline with the shape of 
        [batch_size, num_features * (2**num_subgraph_layers)].

        Args:
            polyline: Polyline, input of SubGraph layer.
                [batch_size, num_vectors (num_past_steps), num_features]
                Inputs should be encoded to the vectors in advance to be used in the Graph Neural Network.

        Returns:
            polyline_features: Features of each polyline.
                [batch_size, num_features * (2**num_subgraph_layers)]
        """
        # 1. Process each polyline through the subgraph layers
        polyline_features = polyline
        batch_size, num_vectors, _ = polyline_features.shape
        for layer in self.polyline_subgraph_layers: 
            # In each layer, the input dimension is num_features * (2**i)
            # The output dimension is num_features * (2**(i+1))
            # [batch_size, num_vectors, input_dim] -> [batch_size, num_vectors, input_dim*2]
            polyline_features = layer(polyline_features)
        # 2. Aggregate the features of each polyline using max pooling
        # [batch_size, num_vectors, num_features * (2**num_subgraph_layers)] ->
        # [batch_size, num_features * (2**num_subgraph_layers)]
        assert polyline_features.shape == (batch_size, num_vectors, self.num_features * (2**self.num_subgraph_layers))
        # [batch_size, num_vectors, num_features * (2**num_subgraph_layers)] ->
        # [batch_size, num_features * (2**num_subgraph_layers), num_vectors]
        y = polyline_features.permute(0, 2, 1)
        # [batch_size, num_features * (2**num_subgraph_layers), num_vectors] ->
        # [batch_size, num_features * (2**num_subgraph_layers)]
        y_max = F.max_pool1d(y, kernel_size=num_vectors)
        # [batch_size, num_features * (2**num_subgraph_layers)] ->
        # [batch_size, num_features * (2**num_subgraph_layers)]
        polyline_features = y_max.squeeze(2)
        assert polyline_features.shape == (batch_size, self.num_features * (2**self.num_subgraph_layers))

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
        self.input_dim = input_dim
        self.node_encoder = MLP(input_dim, input_dim)
        self.edge_encoder = nn.Linear(input_dim * 2, input_dim * 2)

    def forward(self, x: torch.Tensor)->torch.Tensor:
        """
        Forward the inputs of PolylineSubGraphLayer and output the features of each vector.

        Args:
            x: Polyline, input of SubGraph layer.
                [batch_size, num_vectors, input_dim]
            input_dim is variable depending on the layer.
            First layer: input_dim = num_features
            Second layer: input_dim = num_features * 2
            Third layer: input_dim = num_features * 4
            ...
            Last layer: input_dim = num_features * (2**(num_subgraph_layers-1))

        Returns:
            x: Processed features of each vector
                [batch_size, num_vectors, input_dim*2]
        """
        # [batch_size, num_vectors, input_dim] -> [batch_size, num_vectors, input_dim]
        x = self.node_encoder(x)
        # Concat with the max pooling
        x = torch.cat([x, x.max(dim=1)[0].unsqueeze(1)], dim=-1)
        # [batch_size, num_vectors, input_dim*2] -> [batch_size, num_vectors, input_dim*2]
        x = self.edge_encoder(x)

        return x

class GlobalGraph(nn.Module):
    def __init__(self, 
        num_global_layers: int,
        num_global_heads: int,
        emb_dim: int,
    ) -> None:
        """
        Instantiate the GlobalGraph.

        Args:
            num_global_layers: number of layers for the global graph.
            num_global_heads: number of attention heads for the global graph.
            emb_dim: dimension of the input features.
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.num_global_heads = num_global_heads
        # Ensure  embed_dimis divisible by num_heads
        assert emb_dim % num_global_heads == 0
        self.global_graph = nn.ModuleList([
            GlobalGraphLayer(embed_dim=emb_dim, num_heads=num_global_heads) for i in range(num_global_layers)
        ])
        
    def forward(self, 
        polyline_features: torch.Tensor,
        target_index: torch.Tensor,
    ) -> torch.Tensor:
        """Process polyline features through global attention 
        interacting different nodes (=tokens)
        
        Args:
            polyline_features: Polyline features after the subgraph.
                [batch_size, num_polylines, emb_dim]
                NOTE: emb_dim = num_global_graph_input_output_dim = num_features * (2**num_subgraph_layers)

            target_index: Index of the target agent to be predicted, mapping to the batch index.
                [batch_size]
                (ex) target_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,...15] if batch_size = 16. 
                Since VectorNet could only predict trajectory of one agent at a time,
                we need to specify the index of the target agent to be predicted.
                This index is used to select the correct output from the VectorNet.
                
        Returns:
            output_node: Processed global features after attention for each track id.
                [batch_size, emb_dim]
        """
        assert self.emb_dim == polyline_features.shape[3]
        for global_graph_layer in self.global_graph:
            # [batch_size, num_polylines, emb_dim]
            polyline_features = global_graph_layer(polyline_features)
        # Specify the target index if feature.
        assert target_index.shape[0] == polyline_features.shape[0]
        output_node = polyline_features[:, target_index, :]

        return output_node

class GlobalGraphLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        """Processing global multi heaf attentionp
        """
        super().__init__()
        self.W_key = nn.Linear(embed_dim, embed_dim)
        self.W_query = nn.Linear(embed_dim, embed_dim)
        self.W_value = nn.Linear(embed_dim, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, polyline_features: torch.Tensor):
        """
        Process the global graph attention
        
        Args:
            polyline_features: polyline features with the shape of
                [batch_size, num_polylines, emb_dim]
        Returns:
            output
                [batch_size, num_polylines, emb_dim]
        """
        # 1. Generate key, query, and value
        # [batch_size, num_polylines, emb_dim]
        key = self.W_key(polyline_features)
        query = self.W_query(polyline_features)
        value = self.W_value(polyline_features)
        # 2. Process global attention
        output, _ = self.attention(query, key, value)

        return output




