import joblib
import pandas as pd
import streamlit as st
import numpy as np

encode = joblib.load('labelencoder.pkl')
model = joblib.load('logisticregression.pkl')
onehote = joblib.load('onehote.pkl')
scaler = joblib.load('Standardscaler.pkl')


st.title('Prediction App')
# st.write(scaler.feature_names_in_)

CreditScore = st.number_input('Enter your Credit Score')
Geography = st.selectbox('Select your country',['France','Spain','Germany'])
Gender = st.selectbox('Select your gender',['Male','Female'])
Age = st.number_input('Enter tour Age')
Tenure = st.number_input('How many years have you been with the bank')
Balance = st.number_input('Your bank balance in currency units')
NumOfProducts = st.number_input('Enter the number of products')
HasCrCard = st.number_input('Enter if you have Credit Card card')
IsActiveMember = st.number_input('Enter if you are active member')
EstimatedSalary = st.number_input('Enter the estimate salary')

encoded_value = encode.transform([Gender])[0]

onehote_value = onehote.transform([[Geography]]).toarray().flatten()

a=pd.DataFrame([onehote_value],columns=onehote.get_feature_names_out())
b = pd.DataFrame([{'CreditScore':CreditScore,'Geography':Geography,'Gender':Gender,'Tenure':Tenure,'Balance':Balance,'NumOfProducts':NumOfProducts,
'HasCrCard':HasCrCard,'IsActiveMember':IsActiveMember,'EstimatedSalary':EstimatedSalary,'Encode_Gender':encoded_value}])
data = pd.concat([b,a],axis = 1)
st.write(data)

user_data =np.array([[CreditScore,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary,encoded_value]+list(onehote_value)])
scaled = scaler.transform(user_data)
predicts = model.predict(scaled)
pd = pd.Series(predicts[0])
pr = round(pd,0)
if st.button('predict'):
    st.write(f'predicted price of the given data is {pr[0]}')