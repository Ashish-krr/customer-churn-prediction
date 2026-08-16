import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder , OneHotEncoder , StandardScaler
import pickle
import tensorflow as tf
import streamlit as st

model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
import pickle

# Load the label encoder for gender
f_gender = open('label_for_gender.pkl', 'rb')
f_geo = open('one_hot_for_geography.pkl', 'rb')
f_scaler = open('scaler.pkl', 'rb')

# Load the data from the open files
label_gender = pickle.load(f_gender)
one_hot_for_geography = pickle.load(f_geo)
scaler = pickle.load(f_scaler)




st.title('Bank churn model')

geography = st.selectbox('Geography', one_hot_for_geography.categories_[0])
gender = st.selectbox('Gender', label_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


data = {
    'CreditScore':credit_score,
    'Gender':label_gender.transform(['Male'])[0],
    'Age':age,
    'Tenure':tenure,
    'Balance':balance,
    'NumOfProducts':num_of_products,
    'HasCrCard':has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary

}

data = pd.DataFrame([data])


geo_encoder = one_hot_for_geography.transform([[geography]]).toarray()
geo_encoder = pd.DataFrame(geo_encoder,columns = one_hot_for_geography.get_feature_names_out() )

data = pd.concat([data,geo_encoder],axis=1)

Test = scaler.transform(data)

st.write(model.predict(Test))