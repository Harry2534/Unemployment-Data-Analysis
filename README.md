# Unemployment Rate Forecasting in Australia

## Project Overview
This project focuses on forecasting the unemployment rate in Australia using time series analysis techniques such as moving averages, seasonal decomposition, and ARIMA modeling. The dataset includes unemployment rate data from 2011 to 2024, sourced from the Australian Bureau of Statistics (ABS). Additionally, inflation and interest rate data are integrated into the analysis to provide a more comprehensive economic perspective.

## Features
- **Data Preprocessing**: Cleaning and structuring raw data from multiple sources.
- **Exploratory Data Analysis (EDA)**: Identifying trends, seasonality, and key insights.
- **Time Series Decomposition**: Extracting trend, seasonal, and residual components.
- **ARIMA Modeling**: Fitting an ARIMA(2,2,1) model to predict unemployment trends.
- **Forecast Visualization**: Graphical representation of unemployment trends and confidence intervals.
- **Interactive Dashboard**: Built using Streamlit for real-time data interaction and visualization.

## Data Sources
- **Australian Bureau of Statistics (ABS)**: Unemployment rate data
- **Reserve Bank of Australia (RBA)**: Interest rate data
- **ABS CPI Dataset**: Consumer Price Index (CPI) data

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/unemployment-forecast.git
   cd unemployment-forecast
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit dashboard:
   ```bash
   streamlit run app.py
   ```

## Usage
- The dashboard provides an interactive interface to visualize unemployment trends.
- Users can analyze historical data, explore trends, and view ARIMA-based forecasts.
- The confidence intervals in the forecast indicate the range of possible unemployment rate fluctuations.

## Project Structure
```
├── data/                   # Raw datasets
├── scripts/                # Data preprocessing and model scripts
├── app.py                  # Streamlit dashboard
├── requirements.txt        # Required Python packages
├── README.md               # Project documentation
```

## Key Insights
- The unemployment rate exhibits cyclical behavior with seasonal variations.
- A significant spike in 2020 corresponds to COVID-19-related economic disruptions.
- ARIMA-based forecasts suggest an increasing unemployment trend in 2024.
- Seasonal adjustments improve forecast accuracy.

## Future Improvements
- Incorporate additional economic indicators for better predictive accuracy.
- Experiment with deep learning models for time series forecasting.
- Enhance the Streamlit dashboard with more interactive features.

## Contributors
- **Your Name** (Your Email / GitHub Profile)

## License
This project is licensed under the MIT License.

## Acknowledgments
- Australian Bureau of Statistics (ABS)
- Reserve Bank of Australia (RBA)
- Open-source Python libraries: Pandas, NumPy, Statsmodels, Matplotlib, Streamlit


