
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout


path = kagglehub.dataset_download("yashdevladdha/uber-ride-analytics-dashboard")
df = pd.read_csv(f"{path}/ncr_ride_bookings.csv")


df['Timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df = df.sort_values('Timestamp')


df_time = df.set_index('Timestamp').resample('H').size().reset_index()
df_time.columns = ['Timestamp', 'Ride_Count']


scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_time['Ride_Count'].values.reshape(-1, 1))


def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window, 0])
        y.append(data[i+window, 0])
    return np.array(X), np.array(y)

window_size = 24
X, y = create_sequences(scaled_data, window_size)


X = np.reshape(X, (X.shape[0], X.shape[1], 1))


split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


model = Sequential([
    SimpleRNN(50, activation='relu', input_shape=(window_size, 1), return_sequences=True),
    Dropout(0.2),
    SimpleRNN(50, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)


predicted_counts = model.predict(X_test)
predicted_counts = scaler.inverse_transform(predicted_counts)
actual_counts = scaler.inverse_transform(y_test.reshape(-1, 1))

print(f"Sample Prediction: {predicted_counts[0][0]:.2f} rides | Actual: {actual_counts[0][0]}")