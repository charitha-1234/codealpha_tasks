import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

# Task 1: Iris Flower Classification
df_iris = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/IRIS.csv")
X = df_iris.iloc[:, :-1]
y = LabelEncoder().fit_transform(df_iris.iloc[:, -1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_iris = RandomForestClassifier()
model_iris.fit(X_train, y_train)
preds_iris = model_iris.predict(X_test)
print("Iris Classification Accuracy:", accuracy_score(y_test, preds_iris))