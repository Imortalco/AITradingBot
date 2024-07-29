import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

def calculate_metrics_by_group(group, actual, predicted):
    mae = mean_absolute_error(group[actual], group[predicted])
    mse = mean_squared_error(group[actual], group[predicted])
    rmse = np.sqrt(mse)
    r2 = r2_score(group[actual], group[predicted])
    return pd.Series({'MAE' : mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2})

def plot_predictions(df, actual_column, prediction_column, key_column, time_col):
    unique_ids = df[key_column].unique()
    df[time_col] = pd.to_datetime(df[time_col])
    df[actual_column] = pd.to_numeric(df[actual_column], errors='coerce')
    df[prediction_column] = pd.to_numeric(df[prediction_column], errors='coerce')
    plt.figure(figsize=(15, 15))

    for id in unique_ids:
        group = df[df[key_column] == id]
        plt.plot(group[time_col], group[actual_column], label=f'Actual - {id}')
        plt.plot(group[time_col], group[prediction_column], linestyle='--', label=f'Predicted - {id}')
    
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Actual vs Predicted values for each ID')
    
    plt.legend()
    plt.show()

def compare_predictions(df_actual, df_predicted, actual_column, prediction_column, key_column = 'unique_id', time_col = 'time'):
    df_actual[time_col] = pd.to_datetime(df_actual[time_col], errors='coerce')
    df_predicted[time_col] = pd.to_datetime(df_predicted[time_col], errors='coerce')

    df_merged = pd.merge(df_actual[[key_column, actual_column, time_col]],
                         df_predicted[[key_column, prediction_column, time_col]],
                         on=[key_column, time_col])
    
    metrics_by_group = df_merged.groupby(key_column).apply(lambda group: calculate_metrics_by_group(group,actual_column, prediction_column)).reset_index()
    print(metrics_by_group)
    plot_predictions(df_merged, actual_column, prediction_column, key_column, time_col)