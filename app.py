import streamlit as st
import pandas as pd
import os

#profiling
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report

#ML

with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoML")
    choice = st.radio("Navigation",['Upload','Profiling','ML','Download'])
    st.info("This is an AutoML web app")

if os.path.exists('Sourcefile.csv'):
    df=pd.read_csv('Sourcefile.csv',index_col=None)



if choice=='Upload':
    st.title("Upload Your Data for Modelling!")
    file = st.file_uploader("Upload your Dataset Here")
    if file:
        df = pd.read_csv(file,index_col=None)
        df.to_csv("Sourcefile.csv",index=None)
        st.dataframe(df)


if choice=='Profiling':
    st.title('Automated EDA')
    profile_report = ydata_profiling.ProfileReport(df)
    st_profile_report(profile_report)

if choice=='ML':
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

if choice=='Download':
    if os.path.exists("best_model_reg.pkl"):
        with open("best_model_reg.pkl",'rb') as f:
            st.download_button("Download model",f,'trained_model.pkl')
    else:
        with open("best_model_cla.pkl",'rb') as f:
            st.download_button("Download model",f,'trained_model.pkl')

