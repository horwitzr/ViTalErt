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
from bokeh.models.widgets import Div


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
                        24 hours of the current ICU visit, the risk of VTE is {:.2f} times that." \
                        .format(risk))

    if y_probs >  thresh:
        st.write('It is recommended that you administer prophylaxis and monitor the patient for VTE.')
    else:
        st.write('Administering prophylaxis for this patient is not recommended.')
    return risk

#########################################
#########################################
# Load model
filename_pickle = ('models/model_scaler_logRegr_featsel2020_06_20_2302.pickle')
model_and_scaler = pickle.load(open(filename_pickle, 'rb'))
y_probs_TRAIN_mean = model_and_scaler['y_probs_TRAIN_mean']

# Define threshold. If the probability is above this threshold, encourage prophylaxis
# Below this threshold, do not encourage prophylaxis
thresh = 0.441


st.title('ViTalErt: Risk Monitoring for Venous Thromboembolism in ICU Patients')

filename = None
risk = None

# Create radio buttons in sidebar
option_list = ["Select an example patient", \
               "Upload your own csv file", \
               "Manually enter the patient's characteristics"]
x = st.sidebar.radio('Choose an option', option_list)

# Select pre-defined patient
if x == option_list[0]:
    mypatient = st.selectbox('Select a patient',\
                        ('Patient1', 'Patient2', 'Patient3', \
                        'Patient4', 'Patient5', 'Patient6'))
    filename = 'example_patients/' + mypatient + '.csv'
    X_patient = readAndDisplay(filename)
    risk = doPrediction(model_and_scaler, X_patient)

# Upload your own csv file
elif x == option_list[1]:
    filename = st.file_uploader("Choose a csv file", type='csv')
    if filename is not None:
        X_patient = readAndDisplay(filename)
        risk = doPrediction(model_and_scaler, X_patient)

# Play with the model
else:
    age = st.sidebar.slider('Age', 19, 90, 70)
    admissionweight = st.sidebar.slider('Admission Weight (kg)', 40, 250, 108)
    visitnumber = st.sidebar.slider('ICU Visit Number', 1, 10, 1)
    heartrate = st.sidebar.slider('Heart Rate (bpm)', 40, 170, 88)
    aids = st.sidebar.radio('AIDS?',\
                        ('No', 'Yes'))
    ima = st.sidebar.radio('Internal Mammary Artery Graft?', \
                        ('No', 'Yes'))
    midur = st.sidebar.radio('Heart Attack within 6 Months?',\
                        ('No', 'Yes'))
    oobintubday1 = st.sidebar.radio('Intubated?', ('No', 'Yes'))

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

    risk = doPrediction(model_and_scaler, X_patient)
    X_patient_usedfeats = X_patient[['age', \
                                    'admissionweight', \
                                    'visitnumber', \
                                    'heartrate', \
                                    'aids', \
                                    'ima', \
                                    'midur', \
                                    'oobintubday1']]
if risk != None:
    plt.bar([1,2], [1,risk])
    plt.ylim(0, max(1.05*risk,2))
    plt.ylabel('VTE Risk')
    plt.xticks(ticks=[1,2], labels=['Average \nICU Patient', 'This Patient'])
    plt.title("This Patient's Risk of VTE Relative to \nThat of Average ICU Patient")
    st.pyplot()


# Create bar plot to visualize risk relative to that of average ICU patient
# chart_data = pd.DataFrame([1,risk], columns=["a"])
# plt.bar([1,2], [1,risk])
# st.pyplot()


# Create links to code and slides
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown('[Code](https://github.com/horwitzr/ViTalErt)')
st.markdown('[Slides](https://docs.google.com/presentation/d/1HRnvI72UcO8YPx4yXjEM3I97uKo1pPIq6oHjgnra8pU/edit#slide=id.g89385a3c03_0_110)')
