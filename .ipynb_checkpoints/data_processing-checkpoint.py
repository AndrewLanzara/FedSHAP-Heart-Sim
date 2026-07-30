import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

def data_load():
    """
    For Both:
    Loads in the UCI Heart Disease Data found in the data folder.
    Additionally drops 3 columns since they contain more than 30% of the data.
    Convert the label to binary class.
    For Model:
    Split into train and test sets
    For Shap:
    Splits by location

    Return: {"Model": (train_X, test_X, train_y, test_y),
            "Shap": (Cleveland_data, Hungary_data, Switerland_data, VA_Long_Beach_data)}
    """

    heart_df = pd.read_csv("data\heart_disease_uci.csv")

    # Convert to binary classification
    heart_df['num'] = [1 if x > 0 else 0 for x in heart_df['num']]

    # Drop sparse columns and unrelavent id column
    heart_df.drop(columns = ['ca','thal','slope','id'], inplace = True)
    #heart_df['chol'] = heart_df['chol'].replace(0, np.nan)

    # OneHotEncode categorical categories
    heart_df = pd.get_dummies(heart_df, columns=['restecg','cp'], drop_first=False)

    # Convert boolean/binary columns to float to safely preserve NaN values
    for col in ["fbs", "exang"]:
        heart_df[col] = heart_df[col].astype(float)

    for col in heart_df.select_dtypes(include=['object']).columns:
        heart_df[col] = heart_df[col].astype('category')
    

    X = heart_df.drop(columns = ["num", "dataset"])
    y = heart_df["num"]

    # train and test data for the model
    model_tuple = train_test_split(X,y, test_size = 0.2, random_state=0, stratify=y)

    location_list = []
    hospital_dfs = {name: subset.drop(columns=['dataset']) 
                    for name, subset in heart_df.groupby('dataset')}
    for key, df in hospital_dfs.items():

        X = df.drop(columns = ["num"])

        location_list.append((key,X))
    return {"Model": model_tuple, "Shap": location_list}

def calculate_emds_and_plot():
    """
    Calculates pairwise EMD for a dictionary of client data and plots a lower-triangle heatmap.
    """

    heart_df = pd.read_csv("data\heart_disease_uci.csv")
    client_datasets_dict = {name: subset.drop(columns=['dataset']) 
                    for name, subset in heart_df.groupby('dataset')}
    client_names = list(client_datasets_dict.keys())
    num_clients = len(client_names)
    
    # Create an empty matrix filled with zeros
    emd_matrix = np.zeros((num_clients, num_clients))
    
    # Extract the specific data you want to measure (e.g., X_train or y_train)
    flattened_data = {}
    for name, data_tuple in client_datasets_dict.items():
        
        # Unpack your train_test_split tuple. 
        X_train, X_test, y_train, y_test = data_tuple

        # Looking for feature skew
        target_data = X_train.flatten() 
        
        # Looking for label skew
        #target_data = y_train.flatten()
        
        flattened_data[name] = target_data

    # 2. Calculate the Pairwise EMD
    for i in range(num_clients):
        for j in range(i + 1, num_clients): # Only calculate upper triangle
            name_a = client_names[i]
            name_b = client_names[j]
            
            # Calculate EMD between the two clients
            dist = wasserstein_distance(flattened_data[name_a], flattened_data[name_b])
            
            # Matrix is symmetric (A to B is same as B to A)
            emd_matrix[i, j] = dist
            emd_matrix[j, i] = dist 
            
    # 3. Create a mask to hide the upper triangle
    mask = np.triu(np.ones_like(emd_matrix, dtype=bool))

    # 4. Generate the Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        emd_matrix, 
        mask=mask,          # Apply the mask here
        xticklabels=client_names, 
        yticklabels=client_names, 
        annot=False,        
        cmap="YlOrRd",      
        cbar_kws={'label': "Earth Mover's Distance"}
    )
    
    plt.title("Statistical Heterogeneity (Non-IID) Across All FL Clients")
    plt.show()
    
    return emd_matrix

    
