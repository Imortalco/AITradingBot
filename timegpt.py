import pandas as pd
import os
from dotenv import load_dotenv
from nixtla import NixtlaClient
from binance.spot import Spot 

# Get your API Key at dashboard.nixtla.io
load_dotenv()

# 1. Instantiate the NixtlaClient
nixtla_client = NixtlaClient(api_key = os.getenv('NIXTLA_API_KEY'))

# 2. Read historic electricity demand data 
df = pd.read_csv('https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/electricity-short.csv')

# 3. Forecast the next 24 hours
#fcst_df = nixtla_client.forecast(df, h=24, level=[80, 90])

#print(fcst_df.head())

# 4. Plot your results (optional)
# nixtla_client.plot(df, fcst_df, time_col='ds', target_col='y', level=[80, 90])
#binance shit here
client = Spot()
print(client.klines('BTCUSDT','1d'))
