import pandas as pd
import matplotlib.pyplot as plt


file_path = "Used Car Dataset.csv" 
df = pd.read_csv(file_path)



df.rename(columns={df.columns[0]: "car_id"}, inplace=True)

df = df.rename(columns={
    "car_name": "temp_col",
    "registration_year": "car_name"
})

df = df.rename(columns={
    "temp_col": "registration_year"
})

second_col = df.columns[1]


df[["registration_year", "car_name"]] = df[second_col].str.split(" ", n=1, expand=True)


print("Original shape:", df.shape)


df_clean = df.dropna()

print("After dropping NaN:", df_clean.shape)


df_clean.to_csv("used_car_dataset_clean.csv", index=False)

# ---------------------------
# Visualization 1: Cars by Year
# ---------------------------
plt.figure()
df_clean["registration_year"].value_counts().sort_index().plot(kind="bar")
plt.title("Number of Cars by Year")
plt.xlabel("registration_year")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("cars_by_year.png")
plt.close()

# ---------------------------
# Visualization 2: Mileage vs Price
# ---------------------------
plt.figure()
plt.scatter(df_clean["mileage(kmpl)"], df_clean["price(in lakhs)"])
plt.title("mileage(kmpl) vs price(in lakhs)")
plt.xlabel("mileage(kmpl)")
plt.ylabel("price(in lakhs)")
plt.tight_layout()
plt.savefig("mileage(kmpl)_vs_price(in lakhs).png")
plt.close()

print("Charts saved successfully.")
