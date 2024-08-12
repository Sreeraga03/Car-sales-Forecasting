import pandas as pd
import statsmodels.api as sm
import streamlit as st
from pandas.tseries.offsets import DateOffset

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('car_sales_dataset.csv', index_col='Date', parse_dates=True)

df = load_data()

# Fit the SARIMAX model
@st.cache_resource
def fit_model():
    model = sm.tsa.statespace.SARIMAX(df['Car_Sales'], order=(1,1,1), seasonal_order=(1,1,1,12))
    results = model.fit()
    return results

results = fit_model()

# Forecast function
def forecast_sales(results, input_date):
    last_date = df.index[-1]
    num_steps = (input_date.year - last_date.year) * 12 + (input_date.month - last_date.month)
    
    if num_steps <= 0:
        st.error("The forecast date must be after the last date in the dataset.")
        return None

    forecast = results.get_forecast(steps=num_steps)
    forecast_index = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=num_steps, freq='M')
    forecast_df = pd.DataFrame(index=forecast_index, columns=['forecast'])
    forecast_df['forecast'] = forecast.predicted_mean

    return forecast_df

# Streamlit app
st.title('Car Sales Forecast')
st.subheader('Give inputs from 2024 january onwards ')

input_month = st.selectbox("Select Month", list(range(1, 13)))
input_year = st.number_input("Select Year", min_value=2020, max_value=2030, value=2024)

if st.button('Submit'):
    input_date = pd.Timestamp(f"{input_year}-{input_month}-01")
    forecast_df = forecast_sales(results, input_date)
    
    if forecast_df is not None:
        st.write(forecast_df)
        
        # Plot the forecast
        st.line_chart(forecast_df, use_container_width=True)
