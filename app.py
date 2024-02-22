#Libraries
import streamlit as st
import pandas as pd
import os
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report

#OS check of csv file and loading
if os.path.exists('Sourcefile.csv'):
    df=pd.read_csv('Sourcefile.csv',index_col=None)
#Title
st.title("AutoML-001")
st.text("Try any Dataset for a quick Data Analysis and Machine Learning overview")


#header-nav-bar

with st.sidebar:
    st.title('AUTOPANDAS')
    st.text('Automl with pandas-profiling')
    choice = st.radio("Navigation", ["Data","Profiling","Modelling", "Download"])

#Upload-button

#if choice=='Upload':
st.title("Upload Your Data for Modelling!")
file = st.file_uploader("Upload your Dataset Here")
if file: 
    df = pd.read_csv(file, index_col=None)
        #df.to_csv('dataset.csv', index=None)
    
if choice=='Data':
    st.dataframe(df)
#Profiling-button
if choice=='Profiling':
    st.dataframe(df)
    st.title('Automated EDA')
    profile_report = ydata_profiling.ProfileReport(df)
    st_profile_report(profile_report)
#Profiling-button
if choice=='Profiling':
    st.title('Automated EDA')
    profile_report = ydata_profiling.ProfileReport(df)
    st_profile_report(profile_report)

#ML-Button
if choice=='Modelling':
    st.title('Machine Learning')
    target = st.selectbox("Select the column", df.columns)
    model_type = st.selectbox("Select the Model", ['Classification', 'Regression'])
    if st.button("Train Model"):
        if model_type=='Classification':
            from pycaret.classification import setup, compare_models, pull, save_model
            setup(df,target=target)
            setup_df = pull()
            st.info('ML experiment Classification')
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.info('ML Classification model')
            st.dataframe(compare_df)
            best_model
            save_model(best_model,'best_model_cla')

        else:
            from pycaret.regression import setup, compare_models, pull, save_model
            setup(df,target=target)
            setup_df = pull()
            st.info('ML experiment Regression')
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.info('ML Regression model')
            st.dataframe(compare_df)
            best_model
            save_model(best_model,'best_model_reg')

#Download-button
if choice=='Download':
    if os.path.exists("best_model_reg.pkl"):
        with open("best_model_reg.pkl",'rb') as f:
            st.download_button("Download model",f,'trained_model.pkl')
    else:
        with open("best_model_cla.pkl",'rb') as f:
            st.download_button("Download model",f,'trained_model.pkl')

