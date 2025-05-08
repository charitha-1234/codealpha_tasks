import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
df_sales = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/advertising.csv")
X_sales = df_sales.drop(columns=['Sales'])
y_sales = df_sales['Sales']
X_train, X_test, y_train, y_test = train_test_split(X_sales, y_sales, test_size=0.2, random_state=42)
model_sales = RandomForestRegressor()
model_sales.fit(X_train, y_train)
preds_sales = model_sales.predict(X_test)
print("Sales Prediction MAE:", mean_absolute_error(y_test, preds_sales))
print("Sales Prediction R2 Score:", r2_score(y_test, preds_sales))
