import pandas as pd
import os
from dotenv import load_dotenv
from constants.fileNames import *
from nixtla import NixtlaClient
import matplotlib.pyplot as plt
#from binance.spot import Spot 

# Loading .env values
load_dotenv()

# 1. Instantiate the NixtlaClient
nixtla_client = NixtlaClient(api_key = os.getenv('NIXTLA_API_KEY'))
#2. Load data from csv
btc_data = pd.read_csv(BTC_FILE_NAME)
btc_data['time'] = pd.to_datetime(btc_data['time'], unit='s')

train_btc_data = btc_data[(btc_data['time']>='2018-07-01')&(btc_data['time']<='2022-12-31')]
test_btc_data = btc_data[(btc_data['time']>='2023-01-01')&(btc_data['time']<='2023-12-31')]
# 3. Forecast the next 24 hours
fcst_df = nixtla_client.forecast(df = train_btc_data, 
                                 h=len(test_btc_data), 
                                 level=[80, 90],
                                 freq='B',
                                 model = 'timegpt-1-long-horizon',
                                 time_col='time',
                                 target_col='close')

nixtla_client.plot(train_btc_data, fcst_df, models=['TimeGPT'],  level=[80, 90],id_col="BTC", time_col='time', target_col='close').show()

plt.subplots()
plt.show(block=True)
