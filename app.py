#!/usr/bin/env python
# coding: utf-8

# # Settings, Directory Specs, and Imports

# In[1]:


# 0 = no streamlit
# 1 = test user inputs
# 2 = run in streamlit
streamlit_status = 1
do_plots = 0

#dir_read = '/Users/rachellehorwitz/Documents/ViTalErt/data/filtered02/'
#dir_read = '/Users/rachellehorwitz/Documents/VTAlert/over18_eicu/'

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import pickle

#file_name_pickle_read = 'models/model_2020_06_11_1923_week3.pickle'
#file_name_pickle_read = 'models/model_2020_06_11_1934.pickle'
#########################################
def readAndDisplay(filename):
    X_patient = pd.read_csv(filename)
    st.write(X_patient)
    return X_patient
#########################################
def doPrediction(model_and_scaler, X_test):
    clf = model_and_scaler['model']
    scaler = model_and_scaler['scaler']
    feature_mask = model_and_scaler['feature_mask']
    vars_cont = model_and_scaler['vars_cont']
    vars_categ = model_and_scaler['vars_categ']

    X_test_sc = np.concatenate([scaler.transform(X_test[vars_cont]), \
                             X_test[vars_categ].to_numpy()], axis=1)
    X_test_imp     = X_test_sc[:,feature_mask]
    y_probs = clf.predict_proba(X_test_imp)[:,1]

    st.write('-----')
    st.write('The risk of VTE is: {:.2f}%'.format((100*y_probs[0])))

    if y_probs[0] >  thresh:
        st.write('It is recommended that you administer propylaxis and monitor the patient for VTE.')
    else:
        st.write('Administering propylaxis for this patient is not recommended.')
#########################################
#########################################
# Load model
filename_pickle = ('models/model_scaler_logRegr_featsel2020_06_18_2003.pickle')
model_and_scaler = pickle.load(open(filename_pickle, 'rb'))

# Define threshold
thresh = 0.38


st.title('ViTalErt: Risk Monitoring for Venous Thromboembolism in ICU Patients')
st.write('Select an option from the sidebar on the left.')

filename = None

option_list = ['Select an example patient', 'Upload your own csv file']

x = st.sidebar.radio('Choose an option', option_list)
if x == option_list[0]:
    mypatient = st.selectbox('Select a patient',('Patient1', 'Patient2'))
    filename = mypatient + '.csv'
    X_patient = readAndDisplay(filename)
    doPrediction(model_and_scaler, X_patient)
elif x == option_list[1]:
    filename = st.file_uploader("Choose a csv file", type='csv')
    if filename is not None:
        X_patient = readAndDisplay(filename)
        doPrediction(model_and_scaler, X_patient)
else:
    st.write('future is coming')
