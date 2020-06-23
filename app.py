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
    y_probs_TRAIN_mean = model_and_scaler['y_probs_TRAIN_mean']

    X_test_sc = np.concatenate([scaler.transform(X_test[vars_cont]), \
                             X_test[vars_categ].to_numpy()], axis=1)
    X_test_imp     = X_test_sc[:,feature_mask]
    y_probs = clf.predict_proba(X_test_imp)[:,1]
    y_probs = y_probs[0]

    risk = y_probs/y_probs_TRAIN_mean
    st.write('-----')
    st.write("Relative to the average adult ICU patient not \
                        diagnosed with VTE upon admission or within the first \
                        24 hours of the current ICU visit, the risk of VTE is {:.2f} times greater." \
                        .format(risk))

    if y_probs >  thresh:
        st.write('It is recommended that you administer propylaxis and monitor the patient for VTE.')
    else:
        st.write('Administering propylaxis for this patient is not recommended.')
#########################################
#########################################
# Load model
filename_pickle = ('models/model_scaler_logRegr_featsel2020_06_20_2302.pickle')
model_and_scaler = pickle.load(open(filename_pickle, 'rb'))

# Define threshold
thresh = 0.441



st.title('ViTalErt: Risk Monitoring for Venous Thromboembolism in ICU Patients')

filename = None

option_list = ["Select an example patient", \
               "Upload your own csv file", \
               "Manually enter the patient's characteristics"]

x = st.sidebar.radio('Choose an option', option_list)
if x == option_list[0]:
    mypatient = st.selectbox('Select a patient',\
                        ('Patient1', 'Patient2', 'Patient3', \
                        'Patient4', 'Patient5', 'Patient6'))
    filename = 'example_patients/' + mypatient + '.csv'
    X_patient = readAndDisplay(filename)
    doPrediction(model_and_scaler, X_patient)
elif x == option_list[1]:
    filename = st.file_uploader("Choose a csv file", type='csv')
    if filename is not None:
        X_patient = readAndDisplay(filename)
        doPrediction(model_and_scaler, X_patient)
else:
    age = st.slider('Age', 19, 90, 50)
    admissionweight = st.slider('Admission Weight (kg)', 40, 250, 75)
    visitnumber = st.slider('ICU Visit Number', 1, 10, 1)
    heartrate = st.slider('Heart Rate (bpm)', 40, 170, 70)
    aids = st.radio('AIDS?',\
                        ('No', 'Yes'))
    ima = st.radio('Internal Mammary Artery Graft?', \
                        ('No', 'Yes'))
    midur = st.radio('Heart Attack within 6 Months?',\
                        ('No', 'Yes'))
    oobintubday1 = st.radio('Intubated?', ('No', 'Yes'))

    d   = {'age': age, \
            'admissionweight': admissionweight, \
            'admissionheight': 0, \
            'bmi': 0, \
            'gender_Female': 0, \
            'ethnicity_African American': 0, \
            'ethnicity_Asian': 0, \
            'ethnicity_Caucasian': 0, \
            'ethnicity_Hispanic': 0, \
            'ethnicity_Native American': 0, \
            'ethnicity_Other/Unknown': 0, \
            'verbal': 0, \
            'motor': 0, \
            'eyes': 0, \
            'thrombolytics': 0, \
            'aids': aids, \
            'hepaticfailure': 0, \
            'lymphoma': 0, \
            'metastaticcancer': 0, \
            'leukemia': 0, \
            'immunosuppression': 0, \
            'cirrhosis': 0, \
            'activetx': 0, \
            'ima': ima, \
            'midur': midur, \
            'oobventday1': 0, \
            'oobintubday1': oobintubday1, \
            'diabetes': 0, \
            'visitnumber': visitnumber, \
            'heartrate': heartrate
            }
    X_patient = pd.DataFrame({k: [v] for k, v in d.items()})
    X_patient = X_patient.replace('Yes', 1)
    X_patient = X_patient.replace('No', 0)

    doPrediction(model_and_scaler, X_patient)
    X_patient_usedfeats = X_patient[['age', \
                                    'admissionweight', \
                                    'visitnumber', \
                                    'heartrate', \
                                    'aids', \
                                    'ima', \
                                    'midur', \
                                    'oobintubday1']]
