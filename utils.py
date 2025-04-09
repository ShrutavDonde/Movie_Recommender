import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_ratings(filepath):
    return pd.read_csv(filepath)

def load_movies(filepath):
    return pd.read_csv(filepath)

def train_test_split_df(df, test_size=0.2):
    return train_test_split(df, test_size=test_size, random_state=42)