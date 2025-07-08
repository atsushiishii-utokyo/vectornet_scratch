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
   - A batch of polylines, where each polyline is a sequence of vectors (e.g., positions, attributes).
   - Masks indicating valid vectors within each polyline.

2. **PolylineSubGraph:**  
   - Encodes each polyline independently using MLPs and feature aggregation.

3. **GlobalGraph:**  
   - Applies attention or message passing across all polyline features to model their interactions.

4. **Decoder:**  
   - Maps the global features to the desired output, such as future positions of the target agent.

## Usage

The main model is implemented in `network/vectornet.py` as the `VectorNet` class. It can be instantiated and used as follows:

```python
model = VectorNet(input_dim=2, hidden_dim=64, output_dim=2)

# Example input

polylines = torch.randn(batch_size, num_polylines, num_vectors, input_dim)
masks = torch.ones_like(polylines)

# Forward pass

predictions = model(polylines, masks)
```

## Additional Components

### MLP Layer

The `MLP` class in `network/mlp.py` implements a simple Multi-Layer Perceptron with layer normalization and ReLU activation functions. It is used in the `PolylineSubGraphLayer` and `GlobalGraph` modules.

```python
mlp = MLP(input_size=128, output_size=64, hidden_size=128)

# Example usage

x = torch.randn(10, 128)
output = mlp(x)
```

### PolylineSubGraphLayer

The `PolylineSubGraphLayer` class in `network/vectornet.py` implements the local graph processing for each polyline. It uses an MLP to encode the polyline features and a permutation-based aggregation mechanism to capture the spatial relationships within the polyline.

```python

    polyline_subgraph = PolylineSubGraphLayer(input_dim=2, hidden_dim=64)

    # Example usage

    polylines = torch.randn(batch_size, num_polylines, num_vectors, input_dim)
    masks = torch.ones_like(polylines)

    local_features = polyline_subgraph(polylines, masks)
```

### GlobalGraph

The `GlobalGraph` class in `network/vectornet.py` implements the global graph processing for all polylines. It uses an MLP to encode the polyline features and a permutation-based aggregation mechanism to capture the spatial relationships within the polyline.

```python

global_graph = GlobalGraph(input_dim=64, hidden_dim=128)

# Example usage

global_features = global_graph(local_features)
```                     

## Decoder

The `Decoder` class in `network/vectornet.py` implements the decoder for the VectorNet model. It uses an MLP to decode the global features to the desired output, such as future positions of the target agent.

```python
decoder = Decoder(input_dim=128, hidden_dim=64, output_dim=2)

# Example usage

predictions = decoder(global_features)
```

## Training

The model can be trained using the `train.py` script.

```bash
python train.py
``` 

## Evaluation

The model can be evaluated using the `evaluate.py` script.

```bash 
python evaluate.py
``` 
