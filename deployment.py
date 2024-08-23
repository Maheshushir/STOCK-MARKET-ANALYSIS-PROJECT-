import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Function to fetch data from Yahoo Finance
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    return data

# Function to predict using LSTM model
def predict_lstm(model, data, scaler, steps, n_timesteps=60):
    last_data = data[-n_timesteps:].reshape(1, n_timesteps, 1)
    future_predictions = []

    for _ in range(steps):
        next_prediction = model.predict(last_data)
        future_predictions.append(next_prediction[0, 0])
        last_data = np.append(last_data[:, 1:, :], np.array(next_prediction).reshape(1, 1, 1), axis=1)

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_predictions
def main():
# Streamlit app
 st.title("RELIANCE STOCK PRICE FORECAST")

# User input for training data range
 st.header('SELECT RANGE TO TRAIN')
 input_date1 = st.date_input('FROM')
 input_date2 = st.date_input('TO')

# Load data from Yahoo Finance 
 data = load_data('RELIANCE.NS', input_date1 , input_date2)
   

# Scaling the data
 scaler = MinMaxScaler(feature_range=(0, 1))
 scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# User input for prediction date
 input_date = st.date_input('MENTION THE DATE FOR ITS PREDICITION:')

# Ensure the input date is in the future
 if input_date <= data.index[-1].date():
    st.error('Please select a date in the future.')
 else:
    # Calculate the number of days to predict
    steps = (input_date - data.index[-1].date()).days
    
    if st.button('Predict'):
        if steps <= 0:
            st.error('Please select a valid future date.')
        else:
            model = load_model('LSTM.h5')  # Load the LSTM model
            prediction = predict_lstm(model, scaled_data, scaler, steps)
            st.write(f'PREDICTION FOR GIVEN DATE {input_date}: {prediction[-1][0]:.2f}')

if __name__=='__main__':
     main() 