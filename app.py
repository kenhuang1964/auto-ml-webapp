import streamlit as st
import pandas as pd
import os
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report


# ML Stuff
from pycaret.classification import setup, compare_models, pull, save_model, load_model, predict_model

with st.sidebar:
    st.image("https://www.tensorflow.org/static/site-assets/images/marketing/learn/learn-hero.svg")
    st.title("AutoML Webapp")
    choice = st.radio("Navigation", ["Upload", "Profiling", "ML", "Download"])
    st.info("This web application allows you to build an automated machine learning model with a few clicks.")

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col = None)

if choice == "Upload":
    st.title("Upload your dataset (csv)")
    file = st.file_uploader("Upload your dataset", type=["csv"])
    if file:
        df = pd.read_csv(file, index_col = None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)

if choice == "Profiling":
    st.title("Exploratory Data Analysis")
    st.write("Automated Data Analysis")
    profile_report = df.profile_report()
    st_profile_report(profile_report)
    
if choice == "ML":
    st.title("Machine Learning")
    target = st.selectbox("Select your target variable", df.columns)
    if st.button("Train model"):
        setup(df, target = target, silent = True)
        setup_df = pull()
        #st.info("This is the ML Experiment settings")
        #st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("ML Experiment Results")
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')

if choice == "Download":
    with open("best_model.pkl", "rb") as f:
        st.download_button("Download the Model", f, "trained_model.pkl")