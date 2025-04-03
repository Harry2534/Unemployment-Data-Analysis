## Unemployment ARIMA Forecasting Project

Overview

This project aims to analyze and forecast unemployment rates using the ARIMA (AutoRegressive Integrated Moving Average) model. The ARIMA model is a powerful statistical tool for time series forecasting, particularly useful when dealing with economic indicators such as unemployment rates.

Features

Data preprocessing and visualization

Time series decomposition

Stationarity testing (ADF Test)

Model selection using ACF and PACF plots

ARIMA model fitting and validation

Forecasting future unemployment trends

Dataset

The dataset used in this project contains historical unemployment rates over a specified period. The data is sourced from reliable economic and labor statistics databases.

Requirements

To run this project, install the following dependencies:

pip install pandas numpy matplotlib statsmodels seaborn

Usage

Load and preprocess the unemployment data.

Perform exploratory data analysis (EDA) to understand trends and seasonality.

Check for stationarity using the Augmented Dickey-Fuller test.

Determine ARIMA parameters (p, d, q) using ACF and PACF plots.

Fit the ARIMA model and evaluate its performance.

Generate unemployment forecasts.

Run the script:

python unemployment_arima.py

Results

The project outputs a visualized analysis of unemployment trends along with predicted values for future periods. The accuracy of the model is evaluated using metrics like RMSE and AIC/BIC.

Author

Harry Parappukkaran

License

This project is licensed under the MIT License.
