import pandas as pd


file_path = "Used Car Dataset.csv"   
df = pd.read_csv(file_path)

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATASET INFO =====")
print(df.info())

print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())
