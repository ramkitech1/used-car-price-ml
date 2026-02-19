import streamlit as st
import pandas as pd


st.set_page_config(
    page_title="Used Car Price Prediction",
    page_icon="",
    layout="wide"
)


st.title(" Used Car Price Prediction Project")


st.write(
    """
    This project analyzes a used car dataset and builds a Machine Learning model 
    to predict car prices based on features like mileage and manufacturing year.
    Below is a preview of the cleaned dataset used for training.
    """
)


df = pd.read_csv("used_car_dataset_clean.csv")


st.subheader("Dataset Preview (First 50 Rows)")
st.dataframe(df.head(50))
