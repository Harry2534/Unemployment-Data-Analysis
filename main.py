import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from ARIMA import arima_fcst
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA

# Set a modern Matplotlib style
plt.style.use('tableau-colorblind10')



# Streamlit page configuration
st.set_page_config(
    page_title="Exploratory Data Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load and preprocess the data
db = pd.read_excel("6202006.xlsx", 'Data1', header=9).dropna(how='all')

try:
    # Convert the first column to datetime
    db['Date'] = pd.to_datetime(db.iloc[:, 0], format='%b-%Y', errors='coerce')
except Exception as e:
    print(f"Error converting dates: {e}")

# Drop rows with invalid dates and set index
db.dropna(subset=['Date'], inplace=True)
db.set_index('Date', inplace=True)

# Rename and filter relevant columns
db.rename(columns={'A84423620T': 'Unemployment Rate'}, inplace=True)
db = db[['Unemployment Rate']]


# Filter data within the desired date range
Unemployment_Rate = db.loc["2011-01-01":"2023-12-01"]

# Load and preprocess the data
db = pd.read_excel("640101.xlsx", 'Data1', header=9).dropna(how='all')

try:
    # Convert the first column to datetime
    db['Date'] = pd.to_datetime(db.iloc[:, 0], format='%b-%Y', errors='coerce')
except Exception as e:
    print(f"Error converting dates: {e}")

# Drop rows with invalid dates and set index
db.dropna(subset=['Date'], inplace=True)
db.set_index('Date', inplace=True)

# Rename and filter relevant columns
db.rename(columns={'A2325846C': 'CPI'}, inplace=True)
db = db[['CPI']]

# Filter data within the desired date range
Inflation = db.loc["2011-01-01":"2023-12-01"]

# Load and preprocess the data
db = pd.read_excel("f01d.xlsx", 'Data', header=10).dropna(how='all')

try:
    # Convert the first column to datetime
    db['Date'] = pd.to_datetime(db.iloc[:, 0], format='%b-%Y', errors='coerce')
except Exception as e:
    print(f"Error converting dates: {e}")

# Drop rows with invalid dates and set index
db.dropna(subset=['Date'], inplace=True)
db.set_index('Date', inplace=True)

# Rename and filter relevant columns
db.rename(columns={'FIRMMCRTD': 'CRT'}, inplace=True)
db = db[['CRT']]

# Filter data within the desired date range
Interest_Rate = db.loc["2011-01-01":"2023-12-01"]

# Layout: Three columns for insights, visualization, and decomposition
col = st.columns((1.1, 4.5, 3), gap='medium')

with col[0]:
    st.subheader('Insights')

    # Calculate highest and lowest unemployment rates
    highest_value = Unemployment_Rate['Unemployment Rate'].max()
    highest_date = Unemployment_Rate['Unemployment Rate'].idxmax()
    lowest_value = Unemployment_Rate['Unemployment Rate'].min()
    lowest_date = Unemployment_Rate['Unemployment Rate'].idxmin()

    # Display highest and lowest unemployment rates
    st.markdown(f"**🔺 Highest Rate:** {highest_value:.2f}% on {highest_date.strftime('%B %d, %Y')}")
    st.markdown(f"**🔻 Lowest Rate:** {lowest_value:.2f}% on {lowest_date.strftime('%B %d, %Y')}")

    # Compare rates between two specific dates
    earlier_date, latest_date = "2018-12-01", "2023-12-01"
    earlier_rate = Unemployment_Rate.loc[earlier_date, 'Unemployment Rate']
    latest_rate = Unemployment_Rate.loc[latest_date, 'Unemployment Rate']
    change = latest_rate - earlier_rate

    st.markdown(f"**📅 Rate on {earlier_date}:** {earlier_rate:.2f}%")
    st.markdown(f"**📅 Rate on {latest_date}:** {latest_rate:.2f}%")
    st.markdown(f"**📊 Change (2018-2023):** {change:+.2f}%")

    # Add summarized insights
    st.markdown("### Key Insights")
    st.markdown("""
    - The unemployment rate shows cyclical patterns with a sharp spike in 2020 due to COVID-19, followed by a significant decline.
    - Decomposition reveals long-term trends, recurring seasonal effects, and residual spikes due to external shocks like the pandemic.
    - Seasonal patterns emphasize the importance of including adjustments in forecasting models.
    - ARIMA forecasts from 2024 onward capture trends and seasonality, and it suggests that unemployment rates will increase.
    - Confidence intervals for the ARIMA forecast suggests we could potentially see a decline in unemployment rate but it more likely to surge higher.
    """)

with col[1]:
    data1 = Unemployment_Rate.loc['2012-01-01':'2024-12-31']
    data2 = Inflation.loc['2012-01-01':'2024-12-31']
    data3 = Interest_Rate.loc['2012-01-01':'2024-12-31']

    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot Unemployment Rate
    ax.plot(data1.index, data1['Unemployment Rate'], color='#007ACC', linewidth=2, label="Unemployment Rate")

    # Set titles and labels
    ax.set_title('Unemployment Rate', fontsize=16)
    ax.set_ylabel('Rate (%)', fontsize=14)
    ax.set_xlabel('Date', fontsize=14)

    # Add grid and legend
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(frameon=False, fontsize=12)

    # Rotate x-axis labels and adjust layout
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Display the plot
    st.pyplot(fig)

    # Train ARIMA model
    train_data = Unemployment_Rate.loc['2011':'2021'].copy()
    test_data = Unemployment_Rate.loc['2022':'2024'].copy()
    train_data['MA-4'] = train_data['Unemployment Rate'].rolling(12).mean()
    train_data['Unemployment detrended'] = train_data['Unemployment Rate'] - train_data['MA-4']
    raw_monthly_means = train_data.groupby(train_data.index.month)['Unemployment detrended'].mean()
    adjustment = raw_monthly_means.sum() / 12.0
    monthly_means = raw_monthly_means - adjustment
    seasonal = np.tile(monthly_means, int(np.floor(len(train_data) / 12)))
    train_data['Unemployment seasonal'] = seasonal
    train_data['Unemployment residual'] = train_data['Unemployment detrended'] - train_data['Unemployment seasonal']

    # Check if data is stationary
    acf_original = plot_acf(train_data['Unemployment Rate'])
    pacf_original = plot_pacf(train_data['Unemployment Rate'])
    adf_test = adfuller(train_data['Unemployment Rate'])
    print(f'p-value: {adf_test[1]}')
    # data is stationary

    train_data.index = pd.to_datetime(train_data.index)
    train_data = train_data.asfreq('MS')
    endog = train_data["MA-4"]
    arima_2_2_1 = ARIMA(endog, order=(2, 2, 1)).fit()
    print(arima_2_2_1.summary())


    arima_model = ARIMA(endog, order=(2, 2, 1)).fit()
    rima_fcst = arima_2_2_1.get_forecast(steps=24)
    arima_predictions = pd.DataFrame(arima_fcst.predicted_mean)
    arima_predictions.rename(columns={"predicted_mean": "trend"}, inplace=True)
    seasonal_values = monthly_means.tolist() * (len(arima_predictions) // len(monthly_means))
    arima_predictions.loc[:, 'seasonal'] = seasonal_values[:len(arima_predictions)]
    print(arima_predictions.head())

    arima_predictions = arima_predictions.copy()
    arima_predictions.loc[:, 'trend+seasonal'] = arima_predictions['trend'] + arima_predictions['seasonal']
    print(arima_predictions.head())

    arima_fcst.conf_int().head()
    arima_predictions.head()

    arima_predictions = pd.concat([arima_predictions, arima_fcst.conf_int()], axis=1)
    arima_predictions.rename(columns={"lower MA-4": "trend lower CI", "upper MA-4": "trend upper CI"}, inplace=True)
    arima_predictions["seasonal lower CI"] = arima_predictions["trend lower CI"] + arima_predictions['seasonal']
    arima_predictions["seasonal upper CI"] = arima_predictions["trend upper CI"] + arima_predictions['seasonal']

    train_data = Unemployment_Rate.loc['2020':'2024'].copy()

    train_data['MA-4'] = train_data['Unemployment Rate'].rolling(12).mean()
    train_data['Unemployment detrended'] = train_data['Unemployment Rate'] - train_data['MA-4']
    raw_monthly_means = train_data.groupby(train_data.index.month)['Unemployment detrended'].mean()
    adjustment = raw_monthly_means.sum() / 12.0
    monthly_means = raw_monthly_means - adjustment
    seasonal = np.tile(monthly_means, int(np.floor(len(train_data) / 12)))
    train_data['Unemployment seasonal'] = seasonal
    train_data['Unemployment residual'] = train_data['Unemployment detrended'] - train_data['Unemployment seasonal']

    train_data.index = pd.to_datetime(train_data.index)
    train_data = train_data.asfreq('MS')

    endog = train_data["MA-4"]
    arima_2_2_1 = ARIMA(endog, order=(2, 2, 1)).fit()
    print(arima_2_2_1.summary())

    arima_fcst = arima_2_2_1.get_forecast(steps=24)
    arima_predictions = pd.DataFrame(arima_fcst.predicted_mean)
    arima_predictions.rename(columns={"predicted_mean": "trend"}, inplace=True)

    seasonal_values = monthly_means.tolist() * (len(arima_predictions) // len(monthly_means))
    arima_predictions.loc[:, 'seasonal'] = seasonal_values[:len(arima_predictions)]
    print(arima_predictions.head())

    arima_predictions = arima_predictions.copy()
    arima_predictions.loc[:, 'trend+seasonal'] = arima_predictions['trend'] + arima_predictions['seasonal']
    print(arima_predictions.head())

    arima_predictions = pd.concat([arima_predictions, arima_fcst.conf_int()], axis=1)
    arima_predictions.rename(columns={"lower MA-4": "trend lower CI", "upper MA-4": "trend upper CI"}, inplace=True)
    arima_predictions["seasonal lower CI"] = arima_predictions["trend lower CI"] + arima_predictions['seasonal']
    arima_predictions["seasonal upper CI"] = arima_predictions["trend upper CI"] + arima_predictions['seasonal']

    fig, ax_arima_fcst = plt.subplots(figsize=(12, 5))  # Match the size of the second graph

    # Plot the original unemployment rate
    Unemployment_Rate['2015':'2024']['Unemployment Rate'].plot(
        label='Unemployment Rate', color='#007ACC', linewidth=2, ax=ax_arima_fcst
    )

    # Plot the trend+seasonal forecast and its confidence intervals
    arima_predictions['trend+seasonal'].plot(
        label="Trend+Seasonal Forecast", color='orange', linewidth=2, ax=ax_arima_fcst
    )
    ax_arima_fcst.fill_between(
        arima_predictions.index,
        arima_predictions["seasonal lower CI"],
        arima_predictions["seasonal upper CI"],
        color='orange',
        alpha=0.2,
        label="Trend+Seasonal CI"
    )

    # Set the y-axis scale to match the original plot
    ax_arima_fcst.set_ylim(
        Unemployment_Rate['2015':'2024']['Unemployment Rate'].min(),
        Unemployment_Rate['2015':'2024']['Unemployment Rate'].max()
    )

    # Add title, labels, and legend
    ax_arima_fcst.set_title('Unemployment Rate Forecast (2015-2024)', fontsize=16)
    ax_arima_fcst.set_ylabel('Rate (%)', fontsize=14)
    ax_arima_fcst.set_xlabel('Date', fontsize=14)
    ax_arima_fcst.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)  # Add grid for consistency
    ax_arima_fcst.legend(frameon=False, fontsize=12)

    # Adjust x-axis ticks and layout
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Render the plot
    st.pyplot(fig)

# Column 3: Decomposition
with col[2]:
    Unemployment_Rate['MA-4'] = Unemployment_Rate['Unemployment Rate'].rolling(12).mean()
    detrended = Unemployment_Rate['Unemployment Rate'] - Unemployment_Rate['MA-4']
    seasonal = detrended.groupby(detrended.index.month).transform('mean')
    residual = detrended - seasonal

    # Plot decomposition
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    components = [
        ('Original', Unemployment_Rate['Unemployment Rate'], '#1f77b4'),
        ('Trend', Unemployment_Rate['MA-4'], '#FF7F0E'),
        ('Seasonal', seasonal, '#2CA02C'),
        ('Residual', residual, '#D62728'),
    ]

    for ax, (title, data, color) in zip(axes, components):
        data.plot(ax=ax, color=color, linewidth=2)
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Date', fontsize=14)
        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    st.pyplot(fig)
