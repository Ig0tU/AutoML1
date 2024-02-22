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
col1, col2, col3,col4 = st.columns(4)

choice = col1.button("Upload")
choice1 = col2.button("Profiling")
choice2 = col3.button("ML")
choice3 = col4.button("Download")

#Intial page
st.title("Upload Your Data for Modelling!")
file = st.file_uploader("Upload your Dataset Here")
if file:
    df = pd.read_csv(file,index_col=None)
    df.to_csv("Sourcefile.csv",index=None)
    st.dataframe(df)

#Upload-button
if choice:#=='Upload':
    st.title("Upload Your Data for Modelling!")
    file = st.file_uploader("Upload your Dataset Here")
    if file:
        df = pd.read_csv(file,index_col=None)
        df.to_csv("Sourcefile.csv",index=None)
        st.dataframe(df)

#Profiling-button
if choice1:#=='Profiling':
    st.title('Automated EDA')
    profile_report = ydata_profiling.ProfileReport(df)
    st_profile_report(profile_report)

#ML-Button
if choice2: #=='ML':
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
if choice3:#=='Download':
    if os.path.exists("best_model_reg.pkl"):
        with open("best_model_reg.pkl",'rb') as f:
            st.download_button("Download model",f,'trained_model.pkl')
    else:
        with open("best_model_cla.pkl",'rb') as f:
            st.download_button("Download model",f,'trained_model.pkl')

