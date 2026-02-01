

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

!pip install catboost
from catboost import CatBoostClassifier


cat = CatBoostClassifier(iterations=100, learning_rate=0.1, logging_level='Silent', random_state=42)
cat.fit(X_train, y_train)
y_pred_cat = cat.predict(X_test)

print(f"CatBoost Accuracy: {accuracy_score(y_test, y_pred_cat) * 100:.2f}%")