import pandas as pd
import os
from dotenv import load_dotenv
from constants.file_names import *
from nixtla import NixtlaClient
import matplotlib.pyplot as plt
from plot_predictions import compare_predictions

# Loading .env values
load_dotenv()

# 1. Instantiate the NixtlaClient
nixtla_client = NixtlaClient(api_key = os.getenv('NIXTLA_API_KEY'))

#2. Load data from csv
data = pd.read_csv(COIN_FILE_NAME)

train_data = data[(data['time']>='2018-07-01')&(data['time']<='2023-05-31')]

test_data = data[(data['time']>='2023-06-01')&(data['time']<='2023-12-31')]

#region NEW FORECAST
#NOTE: This is commented so that we don`t call the Nixta API everytime we test the script, in order to get fresh data -> uncomment
# 3. Forecast the next 6 months
# fcst_df = nixtla_client.forecast(df = train_data, 
#                                 h=214, 
#                                 freq='D',
#                                 level=[80,90],
#                                 model = 'timegpt-1-long-horizon',
#                                 id_col='unique_id',
#                                 time_col='time',
#                                 target_col='close')

## Saving the new result in the file
# if not os.path.exists(PREDICTIONS_FOLDER):
#     os.makedirs(PREDICTIONS_FOLDER)
# fcst_df.to_csv(PREDICTIONS_FILE_NAME, index=False)
#endregion

#region READING OLD FORECAST FROM CSV
#NOTE: Either read the data from the csv file OR from the result 
fcst_df = pd.read_csv(PREDICTIONS_FILE_NAME)
#endregion

print(fcst_df.head())
compare_predictions(test_data, fcst_df,'close','TimeGPT')

