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


#
st.title('ViTalErt: Risk Monitoring for Venous Thromboembolism in ICU Patients')

patientName= st.radio("Patient Number", ['Patient1', 'Patient2'])
st.write('You selected ' + patientName)
st.write('-----')

X_patient = pd.read_csv(patientName + '.csv')
st.write('Age: ' + str(X_patient.iloc[0]['age']))
st.write('Admission Weight: ' + str(X_patient.iloc[0]['admissionweight']) + ' kg')
st.write('Admission Height: ' + str(X_patient.iloc[0]['admissionheight']) + ' cm')
st.write('Admission Weight: ' + str(X_patient.iloc[0]['admissionweight']) + ' kg')
st.write('BMI: {:.2f} kg/m^2'.format(X_patient.iloc[0]['bmi']))

if X_patient.iloc[0]['gender_Female']==1:
    st.write('Gender: Female')
else:
    st.write('Gender: Male')

if X_patient.iloc[0]['ethnicity_African American']==1:
    st.write('Ethnicity: African American')
elif X_patient.iloc[0]['ethnicity_Asian']==1:
    st.write('Ethnicity: Asian')
elif X_patient.iloc[0]['ethnicity_Caucasian']==1:
    st.write('Ethnicity: Caucasian')
elif X_patient.iloc[0]['ethnicity_Hispanic']==1:
    st.write('Ethnicity: Hispanic')
elif X_patient.iloc[0]['ethnicity_Native American']==1:
    st.write('Ethnicity: Native American')
elif X_patient.iloc[0]['ethnicity_Other/Unknown']==1:
    st.write('Ethnicity: Other/Unknown')
else:
    st.write('Error in ethnicity')

if X_patient.iloc[0]['unitstaytype_admit']==1:
    st.write('Unit Stay Type: Admit')
elif X_patient.iloc[0]['unitstaytype_readmit']==1:
    st.write('Unit Stay Type: Re-Admit')
elif X_patient.iloc[0]['unitstaytype_transfer']==1:
    st.write('Unit Stay Type: Transfer')

st.write('Verbal: ' + str(X_patient.iloc[0]['verbal']))
st.write('Motor: ' + str(X_patient.iloc[0]['motor']))
st.write('Eyes: ' + str(X_patient.iloc[0]['eyes']))
st.write('Thrombolytics: ' + str(bool(X_patient.iloc[0]['thrombolytics'])))
st.write('AIDS: ' + str(bool(X_patient.iloc[0]['aids'])))
st.write('Hepatic Failure: ' + str(bool(X_patient.iloc[0]['hepaticfailure'])))
st.write('Lymphoma: ' + str(bool(X_patient.iloc[0]['lymphoma'])))
st.write('Metastitic Cancer: ' + str(bool(X_patient.iloc[0]['metastaticcancer'])))
st.write('Leukemia: ' + str(bool(X_patient.iloc[0]['leukemia'])))
st.write('Immunosuppression: ' + str(bool(X_patient.iloc[0]['immunosuppression'])))
st.write('Cirrhosis: ' + str(bool(X_patient.iloc[0]['cirrhosis'])))
st.write('Active Treatment: ' + str(bool(X_patient.iloc[0]['activetx'])))
st.write('Internal Mammary Artery Graft: ' + str(bool(X_patient.iloc[0]['ima'])))
st.write('MI within 6 months: ' + str(bool(X_patient.iloc[0]['midur'])))
st.write('Ventilated: ' + str(bool(X_patient.iloc[0]['oobventday1'])))
st.write('Intubated: ' + str(bool(X_patient.iloc[0]['oobintubday1'])))
st.write('Diabetes: ' + str(bool(X_patient.iloc[0]['diabetes'])))
st.write('Visit Number: ' + str(X_patient.iloc[0]['visitnumber']))
st.write('Average HR on Day 1 of ICU: {:.1f}'.format(X_patient.iloc[0]['heartrate']))





model_name_pickle_read = ('models/model_2020_06_11_2128.pickle')
scaler_name_pickle_read = ('models/scaler_2020_06_11_2128.pickle')

clf = pickle.load(open(model_name_pickle_read, 'rb'))
scaler = pickle.load(open(scaler_name_pickle_read, 'rb'))

X_test_sc = scaler.transform(X_patient)
y_probs = clf.predict_proba(X_test_sc)[:,1]

st.write('-----')
st.write('The risk of VTE is: {:.2f}%'.format((100*y_probs[0])))
# if streamlit_status == 1:
#     age = 30
#     admissionweight = 50
#     admissionheight = 90
#     bmi = 35
#     gender_Female = 0
#     eth_AA = 0
#
# if streamlit_status == 2:
#     age = st.slider('Age', 19, 90)
#     admissionweight = st.slider('Admission Weight (kg)', 40, 400)
#
# if (streamlit_status == 1) | (streamlit_status == 2):
#     input_data = {'age': [age], 'admissionweight': [admissionweight], \
#     'admissionheight': [admissionheight], 'bmi': [bmi], \
#     'gender_Female': [gender_Female], 'ethnicity_African American': [eth_AA]}
#     X_test = pd.DataFrame(input_data, columns=['age', 'admissionweight', \
#     'admissionheight', 'bmi', 'gender_Female', 'ethnicity_African American'])
#
#
#
# # Open model
# if (streamlit_status == 1) | (streamlit_status == 2):
#     logisticRegr_sc = pickle.load(open(file_name_pickle_read, 'rb'))
#
# predictions_sc = logisticRegr_sc.predict(X_test)
# prob_sc = logisticRegr_sc.predict_proba(X_test)[:,1]
# # #lprob_sc = logisticRegr_sc.predict_log_proba(X_test)[:,1]
# # if streamlit_status == 0:
# #     scores_sc = logisticRegr_sc.score(X_test, y_test)
# #     print(scores_sc)
# #     print('prob_sc:  min = ' + str(np.min(prob_sc)))
# #     print('\t    max = ' + str(np.max(prob_sc)))
#
# # In[14]:
#
#
#
#
#
# # In[15]:
#
#
# if streamlit_status == 0:
#
#     # Print baseline accuracy
#     N0_bl = patient[patient['label']==0].shape[0]
#     N1_bl = patient[patient['label']==1].shape[0]
#     print('{:d} patients in negative class'.format(N0_bl))
#     print('{:d} patients in positive class'.format(N1_bl))
#     print('If you predict 0 all the time, accuracy is {:.5f}%'.format(N0_bl/(N0_bl+N1_bl)))
#
#     cm = confusion_matrix(list(y_test), predictions_sc)
#     plot_confusion_matrix(logisticRegr_sc, X_test, list(y_test))
#     plot_confusion_matrix(logisticRegr_sc, X_test, list(y_test),  normalize='true')
#
# if streamlit_status == 1:
#     print('The risk of VTE is: {:.2f}%'.format((100*prob_sc[0])))
#
# if streamlit_status == 2:
#     st.write('The risk of VTE is: {:.2f}%'.format((100*prob_sc[0])))
#
