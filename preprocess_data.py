import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def preprocess_data(df):
    # Verify that required columns exist
    if 'time' not in df.columns or 'unique_id' not in df.columns:
        raise KeyError("Columns 'time' and 'unique_id' must be present in the DataFrame")

    # Convert 'time' to datetime format
    df['time'] = pd.to_datetime(df['time'], errors='coerce')

    # Handle missing values in numerical columns
    df.replace(0, np.nan, inplace=True)
    imputer = SimpleImputer(strategy='mean')
    numerical_cols = ['high', 'low', 'open', 'volumefrom', 'volumeto', 'close']
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

    # Scale numerical features
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Ensure there are no duplicate rows in terms of 'unique_id' and 'time'
    if df.duplicated(subset=['unique_id', 'time']).any():
        df = df.drop_duplicates(subset=['unique_id', 'time'])

    return df

