

import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


df = pd.read_csv('/content/sample_data/california_housing_train.csv')


df = df.dropna()

X = df[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']]
y = (df['median_house_value'] > df['median_house_value'].median()).astype(int) 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


svm_model = SVC(kernel='rbf', C=1.0)
svm_model.fit(X_train, y_train)

print(f"SVM Accuracy: {svm_model.score(X_test, y_test):.2f}")
