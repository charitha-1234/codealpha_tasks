import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df_unemployment = pd.read_csv(r"C:\Users\DELL\OneDrive\Documents\Desktop\code alpa\unemployment.csv")

# Remove extra spaces from column names
df_unemployment.columns = df_unemployment.columns.str.strip()

# Print column names for verification
print("Column Names:", df_unemployment.columns.tolist())

# Rename column properly
df_unemployment.rename(columns={"Estimated Unemployment Rate (%)": "Unemployment_Rate"}, inplace=True)

# Remove non-numeric characters (e.g., % symbols) from "Unemployment Rate"
df_unemployment['Unemployment_Rate'] = df_unemployment['Unemployment_Rate'].astype(str).str.replace(r'[^0-9.]', '', regex=True)
df_unemployment['Unemployment_Rate'] = pd.to_numeric(df_unemployment['Unemployment_Rate'], errors='coerce')

# Convert "Date" column to datetime format
df_unemployment['Date'] = df_unemployment['Date'].astype(str).str.strip()  # Remove spaces
df_unemployment['Date'] = pd.to_datetime(df_unemployment['Date'], format="%d-%m-%Y", errors='coerce')

# Drop missing values
df_unemployment.dropna(subset=['Date', 'Unemployment_Rate'], inplace=True)

# Debugging: Check processed data
print("\nProcessed Data Sample:\n", df_unemployment[['Date', 'Unemployment_Rate']].head(10))
print("\nTotal Rows After Cleaning:", df_unemployment.shape[0])

# Ensure data is available for plotting
if df_unemployment.shape[0] == 0:
    print("Error: No valid data available for plotting! Check dataset formatting.")
    exit()

# Plot Unemployment Rate Over Time
plt.figure(figsize=(10, 5))
sns.lineplot(data=df_unemployment, x="Date", y="Unemployment_Rate", marker="o", linestyle="-")
plt.title("Unemployment Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
