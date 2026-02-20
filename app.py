import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="Used Car Price Prediction",
    page_icon="",
    layout="wide"
)


df = pd.read_csv("used_car_dataset_clean.csv")
model = joblib.load("linear_regression_model.pkl")

df.columns = df.columns.str.strip().str.lower()


st.title(" Used Car Price Prediction App")

st.write(
    "Enter vehicle details below to estimate the resale price using a trained Linear Regression model."
)


st.subheader("Enter Car Details")

col1, col2 = st.columns(2)

with col1:
    mileage = st.number_input(
        "Mileage (kms driven)",
        min_value=0,
        max_value=int(df["kms_driven"].max()),
        value=50000,
        step=1000
    )

with col2:
    year = st.selectbox(
        "Manufacturing Year",
        sorted(df["registration_year"].unique(), reverse=True)
    )


if st.button("Predict Price"):
    input_data = pd.DataFrame([[mileage, year]], columns=["kms_driven", "registration_year"])
    prediction = model.predict(input_data)[0]

    st.success(f"Estimated Car Price: ₹ {prediction:,.0f}")


st.subheader("Dataset Preview")
st.dataframe(df.head(50))