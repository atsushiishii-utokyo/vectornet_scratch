# VectorNet

VectorNet is a neural network architecture designed for trajectory prediction in autonomous driving and related domains. It models the spatial and temporal interactions between agents (such as vehicles and pedestrians) and the map environment by representing their motion histories and map elements as sets of polylines (sequences of vectors). VectorNet processes these polylines using a hierarchical graph neural network to capture both local and global context.

## Key Components

- **Polyline Subgraph (Local Graph):**  
  Each polyline (e.g., an agent's trajectory or a map lane) is processed independently to extract local features. This is achieved using a Multi-Layer Perceptron (MLP) and aggregation mechanisms to encode the structure and relationships within each polyline.

- **Global Graph:**  
  The local features from all polylines are then processed together using a global graph module, which models the interactions between different polylines (agents and map elements). This is typically implemented using attention mechanisms or message passing neural networks.

- **Decoder:**  
  The aggregated global features are decoded to produce the final trajectory predictions for the target agent.

## Model Architecture

1. **Input:**  
   - A list of polylines, where each polyline is a sequence of vectors (e.g., positions, attributes).
   - Target agent index for batch processing.
   - Optional masks indicating valid vectors within each polyline.

2. **PolylineSubGraph:**  
   - Encodes each polyline independently using multiple layers of MLPs and feature aggregation.
   - Processes polylines in target agent-centric coordinates.

3. **GlobalGraph:**  
   - Applies multi-head attention across all polyline features to model their interactions.
   - Uses target agent index to select relevant features for prediction.

4. **Trajectory Decoder:**  
   - Maps the global features to future trajectory predictions using an MLP.

## Configuration

The model uses a configuration system defined in `network/config.py`:

```python
VECTORNET_CONFIG = {
    "num_subgraph_layers": 3,        # Number of polyline subgraph layers
    "num_global_layers": 1,          # Number of global graph layers
    "num_global_heads": 3,           # Number of attention heads
    "num_features": 6,               # Vector feature dimension (ds_x,ds_y,de_x,de_y,a,j)
    "num_future_steps": 30,          # Number of future timesteps to predict
    "num_prediction_features": 2,    # Output features (x,y coordinates)
    "num_past_steps": 60,            # Number of past timesteps
    "num_vectors": 60,               # Number of vectors per polyline
}
```

## Usage

### VectorNet_prediction (Recommended)

The main prediction model that combines VectorNet with trajectory decoding:

```python
from network.vectornet import VectorNet_prediction
from network.config import VECTORNET_CONFIG

# Create model
model = VectorNet_prediction(VECTORNET_CONFIG)

# Example input
batch_size = 16
num_polylines = 10
num_vectors = 60
num_features = 6

# Create polylines list
polylines_list = [
    torch.randn(num_vectors, num_features) for _ in range(num_polylines)
]

# Target agent indices
target_index = torch.arange(batch_size)

# Forward pass
predictions = model(polylines_list, target_index)
# Output shape: [batch_size, num_future_steps, num_prediction_features]
```

### VectorNet (Core Architecture)

The core VectorNet architecture without trajectory decoding:

```python
from network.vectornet import VectorNet

# Create model
vectornet = VectorNet(
    num_vectors=60,
    num_features=6,
    num_subgraph_layers=3,
    num_global_layers=1,
    num_global_heads=3
)

# Forward pass
global_features = vectornet(polylines_list, target_index)
# Output shape: [batch_size, global_graph_input_output_dim]
```

## Additional Components

### MLP Layer

The `MLP` class in `network/mlp.py` implements a Multi-Layer Perceptron with layer normalization and ReLU activation functions:

```python
from network.mlp import MLP

mlp = MLP(input_size=128, output_size=64, hidden_size=128)

# Example usage
x = torch.randn(10, 128)
output = mlp(x)
```

### PolylineSubGraph

The `PolylineSubGraph` class processes each polyline independently through multiple layers:

```python
from network.vectornet import PolylineSubGraph

polyline_subgraph = PolylineSubGraph(
    num_subgraph_layers=3,
    num_features=6
)

# Example usage
polyline = torch.randn(batch_size, num_vectors, num_features)
local_features = polyline_subgraph(polyline)
```

### GlobalGraph

The `GlobalGraph` class implements multi-head attention across all polylines:

```python
from network.vectornet import GlobalGraph

global_graph = GlobalGraph(
    num_global_layers=1,
    num_global_heads=3,
    emb_dim=48  # num_features * (2**num_subgraph_layers)
)

# Example usage
polyline_features = torch.randn(batch_size, num_polylines, emb_dim)
global_features = global_graph(polyline_features, target_index)
```

## Testing

Run unit tests to verify the implementation:

```bash
python unit_tests.py
```

The tests cover:
- MLP output shape and type validation
- Gradient computation
- Forward pass consistency

## Development

### Jupyter Notebook

A Jupyter notebook (`vectornet.ipynb`) is provided for interactive development and experimentation.

### Main Entry Point

The `main.py` file serves as the entry point for the application.

## Key Features

- **Target Agent-Centric Processing**: Automatically converts polylines to target agent-centric coordinates
- **Batch Processing**: Supports predicting trajectories for multiple agents simultaneously
- **Configurable Architecture**: Easy to modify network parameters through configuration
- **Modular Design**: Clean separation between polyline processing, global attention, and trajectory decoding
- **Type Hints**: Full type annotations for better code maintainability

## Architecture Details

### Input Processing
- Polylines are represented as sequences of 6-dimensional vectors (ds_x, ds_y, de_x, de_y, a, j)
- Each polyline is processed independently through the subgraph layers
- Features are aggregated using permutation-invariant operations

### Global Attention
- Multi-head attention mechanism captures interactions between all polylines
- Target agent index is used to select relevant features for prediction
- Supports both self-attention and cross-attention patterns

### Output Generation
- Trajectory decoder maps global features to future positions
- Configurable number of future timesteps and prediction features
- Output format: [batch_size, num_future_steps, num_prediction_features] 
