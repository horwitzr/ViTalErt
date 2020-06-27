# ViTalErt
The purpose of this project is to predict the risk of an ICU patient developing venous thromboembolism (VTE), a  potentially fatal condition that occurs when a blood clot forms, often in the deep veins of the leg.  It is a problem in hospitalized patients because they are less mobile than the general population. In many hospitals, "high-risk" patients are given prophylaxis for VTE, but the prophylaxis has the potential to cause bleeding. Different hospitals use different protocols to determine whether the prophylaxis should be administered to each patient. Therefore, the goal of this project is to identify a patient's risk of developing VTE so that the medical professional can administer prophylaxis when appropriate to do so.

# Data
The data comes from the [eICU Collaborative Research Database](https://eicu-crd.mit.edu/), which I accessed through Physionet. Although the data is publicly available, in order to access it, you need to obtain approval. Therefore, I have not included the data on github.

# Running the app
To run the app, cd to the location of the repository and type ```streamlit run app.py``` in the terminal. The URL of the app is https://vitalert.herokuapp.com/.

# Running the Model
See requirements.txt for the packages you will need.
You can run the model with ```notebooks/logisticRegr_model_w_featsel_gridsearch.ipynb```. It assumes that you have pre-processed the data. The variable containing the name of the csv file is ```filename_Xy```. The csv file must contain the following columns:


# Slides
The Google slides for the project can be found at https://docs.google.com/presentation/d/1HRnvI72UcO8YPx4yXjEM3I97uKo1pPIq6oHjgnra8pU/edit

# Credits
The StratifiedGroupKFold code was copied and pasted from https://github.com/scikit-learn/scikit-learn/issues/13621
