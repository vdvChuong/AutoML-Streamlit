# Streamlit for webapp
import streamlit as st

# import PyCaret classification
from pycaret.classification import *

import pandas as pd
import numpy as np

# import for EDA
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import os

if os.path.exists('./train.csv'): 
    df = pd.read_csv('train.csv', index_col=None)
if os.path.exists('./test.csv'): 
    test_df = pd.read_csv('test.csv', index_col=None)

with st.sidebar: 
    #st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.image("automl1.png")
    choice = st.radio("Navigation", ["Upload dataset","Profiling dataset (EDA)","ML Modelling", "Download best model"])
    st.info("This AutoML application helps you explore and build ML model for your dataset!")
    st.image("automl2.png")

if choice == "Upload dataset":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your CSV Dataset",type=["csv"])
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('train.csv', index=None)
        st.dataframe(df)

if choice == "Profiling dataset (EDA)": 
    st.title("Exploratory Data Analysis (EDA)")
    profile_df = ProfileReport(df)
    st_profile_report(profile_df)

if choice == "ML Modelling":

    st.title("AutoML Modelling & Predict")
    # Convert Index object to array
    columns_array = df.columns.to_numpy()

    # Add 'Choose the Target Column' to the beginning of the array
    columns_array = np.insert(columns_array, 0, 'Please choose the Target Column...')

    chosen_target = st.selectbox('Choose the Target Column', columns_array)

    if (chosen_target != 'Please choose the Target Column...'):
        
        st.write('You selected:', chosen_target)
        
        # Setup Experiment
        setup(df, target=chosen_target, silent=True)
        setup_df = pull()
        st.subheader('Experiment Description: ')
        st.dataframe(setup_df)

        # Compare models -> Best model
        st.divider()
        best_model = compare_models(include = ['lightgbm','rf', 'lr', 'knn','nb','svm','dt'])
        compare_df = pull()
        st.subheader('Compare Models: ')
        st.dataframe(compare_df)

        # Select the column to find the maximum value
        acc_col = 'Accuracy'

        # Find the hightest accuracy value
        max_acc = compare_df[acc_col].max()
        st.write("Best Model Accuracy on Train dataset : ",max_acc)

        st.write("Best Model on Train dataset : ",best_model)


        st.divider()


        # The best model use for Test dataset
        best_test = predict_model(best_model)
        df_best_test = pull()
        st.write("Best Model on Test dataset : ")
        st.dataframe(df_best_test)

        # Print the accuracy on Test dataset
        test_acc = df_best_test[acc_col][0]
        st.write("Accuracy on Test dataset : ",test_acc)
        
        # Save and print best ML model pipeline
        save_best_model = save_model(best_model, 'best_model')
        st.write("Best ML Pipeline : ",save_best_model)

        st.divider()

        # Upload test data
        st.subheader("Upload New Test Dataset for Prediction")
        test_file = st.file_uploader("Upload Your New Test Dataset",type=["csv"])
        
        if test_file is not None: 
            test_df = pd.read_csv(test_file,index_col=None)
            test_df.to_csv('test.csv', index=False)
            st.write('New TEST dataset:')
            st.dataframe(test_df)

            # Predict on test data
            st.write('Predicted Values:')
            predictions_df = predict_model(best_model, data=test_df)
            st.dataframe(predictions_df)

            # Display the Predicted value 
            # Get the value counts of predicted column
            st.write('Visualize the Predicted Values:')
            value_counts = predictions_df['Label'].value_counts()

            # Display the bar chart using Streamlit
            st.bar_chart(value_counts)
            st.text(value_counts)

            # Download The Prediction Result
            @st.cache_data
            def convert_df(predictions_df):
                return predictions_df.to_csv(index=False).encode('utf-8')

            csv = convert_df(predictions_df)

            st.download_button(
                "Download result as CSV",
                csv,
                "Prediction_Result.csv",
                "text/csv",
                key='download-csv'
            )

if choice == "Download best model":

    st.title("Best model of your dataset")
    col1, col2, col3 , col4, col5 = st.columns(5)

    with col1:
        pass
    with col2:
        pass
    with col4:
        pass
    with col5:
        pass
    with col3 :
        with open('best_model.pkl', 'rb') as f: 
            st.download_button('Download Model', f, file_name="best_model.pkl")