"""
    First Streamlit App ( https://streamlit.io/gallery )
        streamlit run dev/ex_1.py
        st.write() used to display many data types
"""



import pandas as pd
import streamlit as st
import apps.project_functions as pf  # Assuming this module contains necessary functions

# Assuming a function 'run_model' in project_functions that takes model_name as input and returns results
# def run_model(model_name): ...

@st.cache
def load_data(path):  # Downloads file & caches it in memory
    dataset = pd.read_csv(path)
    return dataset

# List of models (you should replace this with the actual model names)
model_names = ['model_nn', 'model_nb', 'model_nbb', 'model_mnb', 'model_knn', 
               'model_rf', 'model_lr', 'model_svm', 'model_svm1', 'model_xgb']

with st.sidebar:
    st.subheader('About the Dashboard')
    # Display your name
    st.write("SHASHANK")

    st.markdown('Streamlit UI for calling App endpoint.')
    
    # Model selection box
    selected_model = st.selectbox("Choose a model", model_names)

    st.header('Example Request')
    query = st.text_input('Query', 'Is there any investment guide for the stock market in India?')
    num_results = st.text_input('Number of Results Returned', 2)
    num_results = int(num_results)
    submitted = st.button("Submit")

st.title('Streamlit App')
st.markdown(""" 
    Streamlit Example Request Dashboard 
""")

st.header('Returned Response')

if submitted:
    # Assuming you have a function to run the selected model and get results
    results = pf.run_model(selected_model)
    st.write("Results from model:", selected_model)
    st.table(results)  # Display results in a table