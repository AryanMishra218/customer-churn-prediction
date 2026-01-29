import pandas as pd
import os

#downloads real bank dataset
url = "https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/Bank%20Churn%20Modelling.csv"

print("Downloading dataset...")
df = pd.read_csv(url)
df.to_csv('data/bank_churn_data.csv', index=False)

print("Dataset downloaded successfully!")
print(f"Total customers: {len(df)}")
print(f"Total features: {len(df.columns)}")
print("\n🔍 First 5 rows of data:")
print(df.head())

      