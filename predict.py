import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas import to_datetime
from datetime import date
import yfinance as yf

START = "2020-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Define a function to load the dataset
# def load_data(ticker):
#     print(f"Downloading data for {ticker} from {START} to {TODAY}...")
#     data = yf.download(ticker, START, TODAY)
#     data.reset_index(inplace=True)
#     return data

# data = load_data('iot')  # Changed 'iot' to a valid ticker 'AAPL' for testing

data = pd.read_csv('iot.csv')

# Create a new dataframe with only the 'Close' column & Convert the dataframe to a numpy array
dataset = data['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data
# Split the data into x_train and y_train data sets
x_train = []
y_train = []

print("Preparing training data...")
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i <= 61:
        print(x_train)
        print(y_train)
        print()

# Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

print("Building the model...")
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

print("Training the model...")
# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=1)

print("Preparing future sequences...")
# Generate sequences for prediction
future_days = 60  # for example, if you want to predict the next 60 days
x_future = []
for i in range(future_days):
    x_future.append(train_data[-(60+i):-i, 0] if i > 0 else train_data[-60:, 0])
x_future = np.array(x_future)
x_future = np.reshape(x_future, (x_future.shape[0], x_future.shape[1], 1))

print("Making predictions...")
# Get the models predicted price values 
predictions = model.predict(x_future)
predictions = scaler.inverse_transform(predictions)

# Convert predictions to a pandas DataFrame
predictions_df = pd.DataFrame(predictions, columns=['Predictions'])
# Create a DataFrame for the history
history_df = pd.DataFrame(data['Close'], columns=['Close'])
# Append predictions to history
all_data = pd.concat([history_df, predictions_df], ignore_index=True)

# Visualize the data
plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(all_data)
plt.legend(['History', 'Predictions'])
plt.show()