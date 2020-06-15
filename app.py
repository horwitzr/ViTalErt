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

# def file_selector(folder_path='app/'):
#     filenames = os.listdir(folder_path)
#     selected_filename = st.selectbox('Select a file', filenames)
#     return os.path.join(folder_path, selected_filename)




#
st.title('ViTalErt: Risk Monitoring for Venous Thromboembolism in ICU Patients')

filename = st.file_uploader("Choose a csv file", type='csv')
# st.write('You selected `%s`' % filename)

# patientName= st.radio("Patient Number", ['Patient1', 'Patient2'])
# st.write('You selected ' + patientName)
# st.write('-----')



#X_patient = pd.read_csv(patientName + '.csv')
if filename is not None:
    X_patient = pd.read_csv(filename)
    st.write(X_patient)

    filename_pickle = ('models/model_scaler_2020_06_13_1921.pickle')
    model_and_scaler = pickle.load(open(filename_pickle, 'rb'))

    clf = model_and_scaler['model']
    scaler = model_and_scaler['scaler']

    X_test_sc = scaler.transform(X_patient)
    y_probs = clf.predict_proba(X_test_sc)[:,1]

    st.write('-----')
    st.write('The risk of VTE is: {:.2f}%'.format((100*y_probs[0])))

    if 100*y_probs[0] >  0.35:
        st.write('It is recommended that you administer propylaxis and monitor the patient for VTE.')
    else:
        st.write('Administering propylaxis for this patient is not recommended')

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
