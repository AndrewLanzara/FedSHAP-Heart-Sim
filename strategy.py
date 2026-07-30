import torch
import copy
import shap

def federated_average(global_model, client_weights_list):
    """
    Takes a list of client state_dicts, averages them, and updates the global model.
    
    Args:
        global_model: The PyTorch model living on the server.
        client_weights_list: A list of state_dicts returned from client training.
    """
    # Get the dictionary structure of the global model
    global_dict = global_model.state_dict()
    
    # Iterate through every layer's weights and biases
    for key in global_dict.keys():
        # Stack the tensors from all clients for this specific parameter
        stacked_weights = torch.stack([client_dict[key] for client_dict in client_weights_list])
        
        # Calculate the mean across the client dimension (
        global_dict[key] = stacked_weights.mean(dim=0)
        
    # Load the newly averaged weights back into the global model
    global_model.load_state_dict(global_dict)
    
    return global_model


def generate_global_shap(global_model, test_loader):
    """
    Generates SHAP values for the global model using the server's test dataloader.
    """
    # 1. Put model in evaluation mode
    global_model.eval()
    
    # 2. Extract the features (X) from the DataLoader to use in SHAP
    all_features = []
    for inputs, labels in test_loader:
        all_features.append(inputs)
    
    # Concatenate all batches into one large tensor
    background_tensor = torch.cat(all_features, dim=0)
    
    # 3. Initialize the SHAP DeepExplainer
    # Use a random subset (e.g., 100 samples) as the background to speed up computation
    background_subset = background_tensor[:100] 
    explainer = shap.DeepExplainer(global_model, background_subset)
    
    # 4. Calculate SHAP values for the dataset
    shap_values = explainer.shap_values(background_tensor)
    
    return explainer, shap_values, background_tensor