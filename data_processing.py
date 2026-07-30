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
    heart_df['chol'] = heart_df['chol'].replace(0, np.nan)

    # Dictionary mapping original Kaggle column names to plain English
    rename_dict = {
        'age': 'Age',
        'sex': 'Sex',
        'trestbps': 'Resting Blood Pressure',
        'cp': 'Chest Pain Type',
        'chol': 'Cholesterol',
        'fbs': 'Blood Sugar Levels',
        'restecg': 'Resting ECG',
        'thalch': 'Maximum Heart Rate',
        'exang': 'Chest Pain During Exercise',
        'oldpeak': 'Exercise-Induced Heart Stress'
    }

    # Apply the rename dictionary to your dataframe
    heart_df = heart_df.rename(columns=rename_dict)

    # OneHotEncode categorical categories
    heart_df = pd.get_dummies(heart_df, columns=['Resting ECG','Chest Pain Type'], drop_first=False)

    # Convert boolean/binary columns to float to safely preserve NaN values
    for col in ['Blood Sugar Levels', 'Chest Pain During Exercise']:
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


    
