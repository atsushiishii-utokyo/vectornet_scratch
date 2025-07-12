# Configuration for GPT
VECTORNET_CONFIG = {
    "num_subgraph_layers": 3,
    "num_global_layers": 1,
    "num_global_heads": 3, # number of attention heads for the global graph
    "num_features": 6, # number of features for the local graph, the length of the vector v (ds_x,ds_y,de_x,de_y,a,j)
    "num_future_steps": 30,
    "num_prediction_features": 2, # output dimension of the VectorNet in the decoder.(x,y) 
    "num_past_steps": 60, # number of past steps to input to the model, this becomes the number of vectors
    "num_vectors": 60, # number of vectors in the local graph, should be the same as num_past_steps
}