import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
df_car = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/CarPrice.csv")
df_car = df_car.select_dtypes(include=[np.number]).dropna()
X_car = df_car.drop(columns=['price'])
y_car = df_car['price']
X_train, X_test, y_train, y_test = train_test_split(X_car, y_car, test_size=0.2, random_state=42)
model_car = RandomForestRegressor()
model_car.fit(X_train, y_train)
preds_car = model_car.predict(X_test)
print("Car Price Prediction MAE:", mean_absolute_error(y_test, preds_car))
print("Car Price Prediction R2 Score:", r2_score(y_test, preds_car))
