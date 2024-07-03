import cryptocompare
import pandas as pd
import os
from constants.fileNames import *

end_date = '2024-01-01'

btc_data = cryptocompare.get_historical_price_day(coin='BTC',currency='USD', limit=2000, toTs=pd.to_datetime(end_date))
df = pd.DataFrame(btc_data)
df.dropna(axis=1,inplace=True)
df.reset_index(drop=True, inplace=True) 
print(df)

if not os.path.exists(CSV_FOLDER):
    os.makedirs(CSV_FOLDER)

df.to_csv(BTC_FILE_NAME, index=False)
