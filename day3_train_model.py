import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib


df = pd.read_csv("used_car_dataset_clean.csv")


df.columns = df.columns.str.strip().str.lower()


X = df[["kms_driven", "registration_year"]]      # Features
y = df["price(in lakhs)"]             # Target


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)

print("Model trained successfully")
print("Mean Absolute Error:", mae)


joblib.dump(model, "linear_regression_model.pkl")

print("Model saved as linear_regression_model.pkl")
