

import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Download and Load Data
path = kagglehub.dataset_download("uciml/iris")

df = pd.read_csv(f"{path}/Iris.csv")


X = df.drop(['Id', 'Species'], axis=1)
y = df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

base_models = [
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=3)),
    ('svc', SVC(probability=True))
]


stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression()
)

final_ensemble = VotingClassifier(
    estimators=[
        ('stacking', stacking_model),
        ('extra_rf', RandomForestClassifier(n_estimators=50))
    ],
    voting='soft' 
)


final_ensemble.fit(X_train, y_train)
y_pred = final_ensemble.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Ensemble Model Accuracy: {accuracy * 100:.2f}%")