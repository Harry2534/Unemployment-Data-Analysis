import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

db = pd.read_excel("6202006.xlsx", 'Data1', header=9)
db = db.dropna(how='all')

try:
    db['Date'] = pd.to_datetime(db.iloc[:, 0], format='%b-%Y', errors='coerce')
except Exception as e:
    print(f"Error converting dates: {e}")

db.dropna(subset=['Date'], inplace=True)

db.set_index('Date', inplace=True)

if 'A84423620T' in db.columns:
    db.rename(columns={'A84423620T': 'Unemployment Rate'}, inplace=True)
    db = db[['Unemployment Rate']]
else:
    print(f"Series ID {'A2325846C'} not found.")

Unemployment_Rate = db.loc["2011-01":"2023-12"]
Unemployment_Rate.info()
Unemployment_Rate.plot(figsize=(24,6))


train_data = Unemployment_Rate.loc['2011':'2021'].copy()
test_data = Unemployment_Rate.loc['2022':'2024'].copy()

train_data['MA-4'] = train_data['Unemployment Rate'].rolling(12).mean()
train_data['Unemployment detrended'] = train_data['Unemployment Rate'] - train_data['MA-4']

raw_monthly_means = train_data.groupby(train_data.index.month)['Unemployment detrended'].mean()
adjustment = raw_monthly_means.sum()/12.0
monthly_means = raw_monthly_means - adjustment
seasonal = np.tile(monthly_means, int(np.floor(len(train_data) / 12)))
train_data['Unemployment seasonal'] = seasonal
train_data['Unemployment residual'] = train_data['Unemployment detrended'] - train_data['Unemployment seasonal']

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_predict
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller


acf_original = plot_acf(train_data['Unemployment Rate'], lags=40)
pacf_original = plot_pacf(train_data['Unemployment Rate'], lags=40)
adf_test = adfuller(train_data['Unemployment Rate'])
print(f'p-value: {adf_test[1]}')
# data is stationary so no need to difference

endog = train_data["MA-4"]
arima_2_2_1 = ARIMA(endog, order=(2, 2, 1)).fit()
print(arima_2_2_1.summary())
fig = plt.figure(figsize=(16, 9))
fig = arima_2_2_1.plot_diagnostics(fig=fig, lags=24)

arima_fcst = arima_2_2_1.get_forecast(steps=24)
arima_predictions = pd.DataFrame(arima_fcst.predicted_mean)
arima_predictions.rename(columns={"predicted_mean": "trend"}, inplace=True)

fig, ax_arima_fcst = plt.subplots(figsize=(24,6))
train_data['MA-4'].plot(label='Original', ax=ax_arima_fcst)
arima_predictions['trend'].plot(label="ARIMA(2,2,1) trend fcst", ax=ax_arima_fcst)
plt.legend()



























