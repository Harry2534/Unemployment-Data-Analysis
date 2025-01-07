import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

st.set_page_config(layout="wide")

st.title("The Impact of Unemployment on Australia's Economy")


col1, col2, col3 = st.columns([1, 5, 1])

with col1:
    pass

with col2:
    st.write("""
    <div style="font-size: 26px;"> 
        Unemployment serves as a vital barometer of a nation's economic health and the stability of its workforce. For policymakers, economists, and stakeholders, it provides essential insights into labour market conditions, shaping decisions on fiscal strategies and monetary policies.
        This Analysis takes a deep dive into Queensland's Labour Force data, exploring historical trends and unveiling a predictive model for forecasting future unemployment rates. By analyzing the evolution of unemployment over time, we aim to shed light on its potential trajectory in the coming months.
        Our findings will equip government officials, economists, and other stakeholders with valuable insights to address labour market challenges and make informed decisions. Stay with us as we uncover the story behind the numbers and chart a path for Queensland's economic future.
    </div>
    """, unsafe_allow_html=True)

with col3:
    pass



def load_and_process_data(file_name, sheet_name, date_col_idx, series_id, new_col_name):
    db = pd.read_excel(file_name, sheet_name=sheet_name, header=9)
    db = db.dropna(how='all')

    try:
        db['Date'] = pd.to_datetime(db.iloc[:, date_col_idx], format='%b-%Y', errors='coerce')
    except Exception as e:
        print(f"Error converting dates: {e}")


    db.dropna(subset=['Date'], inplace=True)

    db.set_index('Date', inplace=True)

    if series_id in db.columns:
        db.rename(columns={series_id: new_col_name}, inplace=True)
        db = db[[new_col_name]]
    else:
        print(f"Series ID {series_id} not found.")


    return db.loc['2011-01-01':'2023-12-01'].resample('QE').mean()


Consumer_Price_Index = load_and_process_data(
    '640101.xlsx',
    sheet_name='Data1',
    date_col_idx=0,
    series_id='A2325846C',
    new_col_name='Consumer Price Index'
)

Unemployment_Rate  = load_and_process_data(
    '6202006.xlsx',
    sheet_name='Data1',
    date_col_idx=0,
    series_id='A84423620T',
    new_col_name='Unemployment Rate'
)


Cash_Rate_Target  = load_and_process_data(
    'f01d.xlsx',
    sheet_name='Data',
    date_col_idx=0,
    series_id='04-Oct-2024',
    new_col_name='Cash Rate Target'
)

combined_quarterly = pd.concat([Unemployment_Rate, Cash_Rate_Target, Consumer_Price_Index], axis=1)

fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)

titles = ['Unemployment Quarterly', 'Cash Rate Quarterly', 'Inflation Quarterly']

for i, column in enumerate(combined_quarterly.columns):
    rolling_mean = combined_quarterly[column].rolling(window=4).mean()
    rolling_std = combined_quarterly[column].rolling(window=4).std()

    axes[i].plot(combined_quarterly.index, combined_quarterly[column], label=column, marker='o', alpha=0.5)
    axes[i].fill_between(combined_quarterly.index, rolling_mean - rolling_std, rolling_mean + rolling_std, alpha=0.2)

    axes[i].set_title(titles[i])
    axes[i].set_ylabel('Values')
    axes[i].legend()
    axes[i].grid()

plt.xlabel('Date')
plt.tight_layout()

col1, col2 = st.columns([1, 1])

with col1:
    st.write("")
    st.write("")
    st.pyplot(fig)

with col2:
    st.markdown("""
        <div style="font-size: 24px;">
            <br><br>                                                                                                                                                                                   
            Unemployment Quarterly: Unemployment fluctuated between approximately 5% and 7% from 2011 to around 2020. After 2020, there was a noticeable spike, with unemployment rising to around 
            7.5%, likely due to the COVID-19 pandemic's economic impact. However, from mid-2021 onwards, there is a sharp decline, bringing unemployment down to around 4%. Rolling Mean and 
            Standard Deviation: The shaded blue area shows the standard deviation, indicating increased volatility around the 2020 spike, followed by stabilization as the unemployment rate 
            decreases post-2021. 
        </div>
    """, unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")

    st.markdown("""
            <div style="font-size: 24px;">
            <br><br>  
                Cash Rate Quarterly: There is a steady decline in the cash rate from around 4.75% in 2011 to nearly 0% by 2020, reflecting the global trend of low interest rates 
                during this period. In 2021-2022, there's a significant shift as rates begin to climb rapidly, reaching around 4% by 2023, in response to inflationary pressures. Rolling Mean and 
                Standard Deviation: The cash rate's decline is very stable from 2011 to 2020 (narrow shaded area), but volatility increases significantly during the recent hike, reflecting uncertainty
                in financial markets as central banks respond to inflation. 

            </div>
        """, unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")
    st.write("")

    st.markdown("""
            <div style="font-size: 24px;">  
            <br><br> 
                Inflation Quarterly: Inflation, represented by the Consumer Price Index (CPI), shows a steady upward trend, with relatively
                moderate growth between 2011 and 2020. However, after 2020, there is a steep rise in inflation, with the CPI increasing significantly, reflecting the post-pandemic inflationary period. 
                Rolling Mean and Standard Deviation: The shaded region indicates relatively low volatility in the earlier years, but from 2020 onwards, there’s a notable increase in both the CPI and 
                its standard deviation, illustrating the sharp rise in inflation. Overall Insights: The unemployment rate and inflation were relatively stable, and the cash rate was on a steady 
                decline pre 2020. Economic disruption is evident, likely triggered by the COVID-19 pandemic post 2020. There’s a spike in unemployment, followed by a recovery, a sharp increase in 
                inflation, and a rise in cash rates to counter inflation.
            </div>
        """, unsafe_allow_html=True)

corr_matrix = combined_quarterly.corr()

fig, ax = plt.subplots(figsize=(8, 6))

cax = ax.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

fig.colorbar(cax)

ax.set_xticks(np.arange(len(corr_matrix.columns)))
ax.set_yticks(np.arange(len(corr_matrix.columns)))

ax.set_xticklabels(corr_matrix.columns, fontsize=6)
ax.set_yticklabels(corr_matrix.columns, rotation=90, fontsize=6)

for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha='center', va='center', color='black')

plt.tight_layout()
st.pyplot(fig)