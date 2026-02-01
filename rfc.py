

import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


path = kagglehub.dataset_download("uciml/iris")
df = pd.read_csv(f"{path}/Iris.csv")


X = df.drop(['Id', 'Species'], axis=1)
y = df['Species']


le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)
y_pred_rfc = rfc.predict(X_test)

print(f"RFC Accuracy: {accuracy_score(y_test, y_pred_rfc) * 100:.2f}%")