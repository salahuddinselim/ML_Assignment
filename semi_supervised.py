
import pandas as pd
import numpy as np
import os
import kagglehub
from sklearn.semi_supervised import LabelSpreading
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


path = kagglehub.dataset_download("kimjihoo/coronavirusdataset")
df = pd.read_csv(os.path.join(path, 'PatientInfo.csv'))


df['target'] = df['state'].apply(lambda x: 1 if x == 'deceased' else 0)
df['age_clean'] = df['age'].astype(str).str.extract('(\d+)').astype(float)
sub_df = df[['age_clean', 'sex', 'target']].dropna().copy()
sub_df['sex_encoded'] = LabelEncoder().fit_transform(sub_df['sex'].astype(str))

X = sub_df[['age_clean', 'sex_encoded']].values
y = sub_df['target'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



rng = np.random.RandomState(42)
y_train_masked = np.copy(y_train)
random_unlabeled_points = rng.rand(len(y_train)) < 0.7
y_train_masked[random_unlabeled_points] = -1


lp_model = LabelSpreading(kernel='knn', n_neighbors=7)
lp_model.fit(X_train_scaled, y_train_masked)


y_pred = lp_model.predict(X_test_scaled)


print("--- Performance Evaluation ---")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Survived/Other', 'Deceased']))