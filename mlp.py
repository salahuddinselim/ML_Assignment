
import kagglehub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score


path = kagglehub.dataset_download("yashdevladdha/uber-ride-analytics-dashboard")

df = pd.read_csv(f"{path}/ncr_ride_bookings.csv")


cols_to_drop = ['Booking ID', 'Customer ID', 'Date', 'Time', 'Pickup Location', 'Drop Location',
                'Reason for cancelling by Customer', 'Driver Cancellation Reason', 'Incomplete Rides Reason']
df = df.drop(columns=cols_to_drop)


for col in df.select_dtypes(include=np.number).columns:
   
    if col != 'Booking Status' and df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())


le = LabelEncoder()
categorical_cols = ['Vehicle Type', 'Payment Method']
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))


df['Booking Status'] = le.fit_transform(df['Booking Status'])


X = df.drop('Booking Status', axis=1)
y = df['Booking Status']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


mlp = MLPClassifier(hidden_layer_sizes=(64, 32),
                    activation='relu',
                    solver='adam',
                    max_iter=300,
                    random_state=42)

mlp.fit(X_train, y_train)


predictions = mlp.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
print("\nClassification Report:\n", classification_report(y_test, predictions))