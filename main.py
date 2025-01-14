import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
import altair as alt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_predict
import statsmodels.api as sm
from streamlit_extras.stylable_container import stylable_container

st.set_page_config(
    page_title="Exploratory Data Analysis",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

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

Unemployment_Rate = db.loc["2011-01-01":"2023-12-01"]



col = st.columns((1.5, 4.5, 2.5), gap='medium')


with col[0]:
    highest_value = Unemployment_Rate['Unemployment Rate'].max()
    highest_date = Unemployment_Rate['Unemployment Rate'].idxmax()

    lowest_value = Unemployment_Rate['Unemployment Rate'].min()
    lowest_date = Unemployment_Rate['Unemployment Rate'].idxmin()

    Unemployment_Rate = Unemployment_Rate.sort_index()
    st.subheader('Unemployment Rate Insights')
    st.markdown(f"**🔺 Highest Unemployment Rate:** {highest_value:.2f}% on {highest_date.strftime('%B %d, %Y')}")
    st.markdown(f"**🔻 Lowest Unemployment Rate:** {lowest_value:.2f}% on {lowest_date.strftime('%B %d, %Y')}")

    latest_date = "2023-12-01"
    earlier_date = "2018-12-01"

    latest_unemployment_rate = Unemployment_Rate.loc[latest_date, 'Unemployment Rate']
    earlier_unemployment_rate = Unemployment_Rate.loc[earlier_date, 'Unemployment Rate']
    unemployment_change = latest_unemployment_rate - earlier_unemployment_rate

    st.markdown(f"**📅 Unemployment Rate on {earlier_date}:** {earlier_unemployment_rate:.2f}%")
    st.markdown(f"**📅 Unemployment Rate on {latest_date}:** {latest_unemployment_rate:.2f}%")

    if unemployment_change > 0:
        st.markdown(f"**📊 Change in Unemployment Rate (2018-2023):** +{unemployment_change:.2f}%")
    else:
        st.markdown(f"**📊 Change in Unemployment Rate (2018-2023):** {unemployment_change:.2f}%")

with col[1]:
    unemployment_filtered = Unemployment_Rate.loc['2012-01-01':'2024-12-31']

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(unemployment_filtered.index, unemployment_filtered['Unemployment Rate'], color='tab:red')
    ax.set_title('Unemployment Rate (2012-2024)', fontsize=20)
    ax.set_ylabel('UR Value', fontsize=16)
    ax.set_xlabel('Date', fontsize=16)
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)
    st.write("")
    st.write("")

    train_data = Unemployment_Rate.loc['2020':'2024'].copy()
    train_data['MA-4'] = train_data['Unemployment Rate'].rolling(12).mean()
    train_data['Unemployment detrended'] = train_data['Unemployment Rate'] - train_data['MA-4']
    raw_monthly_means = train_data.groupby(train_data.index.month)['Unemployment detrended'].mean()
    adjustment = raw_monthly_means.sum() / 12.0
    monthly_means = raw_monthly_means - adjustment
    seasonal = np.tile(monthly_means, int(np.floor(len(train_data) / 12)))
    train_data['Unemployment seasonal'] = seasonal
    train_data['Unemployment residual'] = train_data['Unemployment detrended'] - train_data['Unemployment seasonal']
    trend_d1 = train_data['MA-4'].diff()
    trend_d2 = trend_d1.diff()
    train_data.index = pd.to_datetime(train_data.index)
    train_data = train_data.asfreq('MS')
    endog = train_data["MA-4"]
    arima_2_2_1 = ARIMA(endog, order=(2, 2, 1)).fit()
    # print(arima_2_2_1.summary())
    # fig = plt.figure(figsize=(16, 9))
    # fig = arima_2_2_1.plot_diagnostics(fig=fig, lags=24)
    arima_fcst = arima_2_2_1.get_forecast(steps=24)
    arima_predictions = pd.DataFrame(arima_fcst.predicted_mean)
    arima_predictions.rename(columns={"predicted_mean": "trend"}, inplace=True)
    fig, ax_arima_fcst = plt.subplots(figsize=(24, 6))
    train_data['MA-4'].plot(label='Original', ax=ax_arima_fcst)
    seasonal_values = monthly_means.tolist() * (len(arima_predictions) // len(monthly_means))
    arima_predictions.loc[:, 'seasonal'] = seasonal_values[:len(arima_predictions)]
    arima_predictions = arima_predictions.copy()
    arima_predictions.loc[:, 'trend+seasonal'] = arima_predictions['trend'] + arima_predictions['seasonal']
    fig, ax_arima_fcst = plt.subplots(figsize=(24, 6))
    train_data['MA-4'].plot(label='Original', ax=ax_arima_fcst)
    arima_predictions['trend'].plot(label="ARIMA(2,2,1) trend fcst", ax=ax_arima_fcst)
    arima_predictions['trend+seasonal'].plot(label="Trend+seasonal fcst", ax=ax_arima_fcst)
    fig, ax_arima_fcst = plt.subplots(figsize=(24, 6))
    train_data['MA-4'].plot(label='Original', ax=ax_arima_fcst)
    arima_predictions['trend'].plot(label="ARIMA(2,2,1) trend fcst", ax=ax_arima_fcst)
    arima_predictions['trend+seasonal'].plot(label="Trend+seasonal fcst", ax=ax_arima_fcst)
    arima_predictions = pd.concat([arima_predictions, arima_fcst.conf_int()], axis=1)
    arima_predictions.rename(columns={"lower MA-4": "trend lower CI", "upper MA-4": "trend upper CI"}, inplace=True)
    arima_predictions["seasonal lower CI"] = arima_predictions["trend lower CI"] + arima_predictions['seasonal']
    arima_predictions["seasonal upper CI"] = arima_predictions["trend upper CI"] + arima_predictions['seasonal']
    fig, ax_arima_fcst = plt.subplots(figsize=(24, 8), dpi=100)
    train_data['Unemployment Rate'].plot(label='Actual Unemployment Rate', ax=ax_arima_fcst, color='blue', linewidth=2)
    arima_predictions['trend+seasonal'].plot(ax=ax_arima_fcst, color='orange', label='Predicted', linewidth=2)
    arima_predictions['seasonal upper CI'].plot(ax=ax_arima_fcst, color='grey', linestyle='--',
                                                label='Upper Confidence Interval', linewidth=1)
    arima_predictions['seasonal lower CI'].plot(ax=ax_arima_fcst, color='grey', linestyle='--',
                                                label='Lower Confidence Interval', linewidth=1)

    ax_arima_fcst.fill_between(
        arima_predictions.index,
        arima_predictions['seasonal lower CI'],
        arima_predictions['seasonal upper CI'],
        color='grey',
        alpha=0.2,
        label='Confidence Interval'
    )

    ax_arima_fcst.set_title('ARIMA Forecast: Unemployment Rate with Confidence Intervals', fontsize=40, pad=20)
    ax_arima_fcst.set_xlabel('Date', fontsize=40)
    ax_arima_fcst.set_ylabel('Unemployment Rate', fontsize=40)
    ax_arima_fcst.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax_arima_fcst.legend(loc='lower left', fontsize=12, frameon=True, shadow=True, borderpad=1)
    plt.tight_layout()
    st.pyplot(fig)



with col[2]:
    st.subheader('Decomposition of Unemployment Rate: Trend, Seasonal, and Residual Components')

    Unemployment_Rate['MA-4'] = Unemployment_Rate['Unemployment Rate'].rolling(12).mean()
    Unemployment_Rate['Unemployment-detrended'] = Unemployment_Rate['Unemployment Rate'] - Unemployment_Rate['MA-4']
    raw_monthly_means = Unemployment_Rate.groupby(Unemployment_Rate.index.month)['Unemployment-detrended'].mean()
    adjustment = raw_monthly_means.sum() / 12.0
    monthly_means = raw_monthly_means - adjustment
    seasonal = np.tile(monthly_means, int(np.floor(len(Unemployment_Rate) / 12)))
    Unemployment_Rate['Unemployment-seasonal'] = seasonal
    Unemployment_Rate['Unemployment_residual'] = Unemployment_Rate['Unemployment-detrended'] - Unemployment_Rate[
        'Unemployment-seasonal']

    fig, ax_str = plt.subplots(4, figsize=(16, 9))

    Unemployment_Rate["2019":"2024"]['Unemployment Rate'].plot(label='Original', ax=ax_str[0])
    Unemployment_Rate["2019":"2024"]['MA-4'].plot(color='orange', label='MA-24 Trend', ax=ax_str[1])
    Unemployment_Rate["2019":"2024"]['Unemployment-seasonal'].plot(color='blue', label='Seasonal', ax=ax_str[2])
    Unemployment_Rate["2019":"2024"]['Unemployment_residual'].plot(color='green', label='Residual', ax=ax_str[3])

    fig.legend()
    plt.tight_layout()

    st.pyplot(fig)

    arima_params = arima_2_2_1.params
    pvalues = arima_2_2_1.pvalues
    aic = arima_2_2_1.aic

    st.subheader('ARIMA Model Important Values')
    st.markdown(f"**ARIMA(2, 2, 1) Coefficients:**")
    for param, value in arima_params.items():
        st.markdown(f"- **{param}:** {value:.4f}")

    st.markdown(f"**P-values for the ARIMA model:**")
    for param, p_value in pvalues.items():
        st.markdown(f"- **{param}:** {p_value:.4f}")

    st.markdown(f"**AIC (Akaike Information Criterion):** {aic:.2f}")