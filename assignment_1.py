# -*- coding: utf-8 -*-
#"""Assignment 1

import requests
import subprocess
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#pip install streamlit

import streamlit as st

df= pd.read_csv("card_transdata.csv")

df.head()

df.shape

#Check missing data, no data missing
df.isnull().sum()

#Check data info
df.info()

#Check the legit transaction with the headding "fraud"
# Have 37634 transaction that legit and 3608 is fraud => very unbalanced

df['fraud'].value_counts()

# Start Analysing

#Seperate
Legit_trans= df[df.fraud == 0]
Fraud_trans = df[df.fraud == 1]

Legit_trans.distance_from_home.describe()

Fraud_trans.distance_from_home.describe()

#Corrected stored in the variables
#Now check compare the value of both type of transaction to get the logic to train the machine

df.groupby('fraud').mean()

#create sample to train data
legit_sample=Legit_trans.sample(n=3608)

#concatrating sample and fraudulent data frame
new_dataset=pd.concat([legit_sample,Fraud_trans], axis=0 )

#Check data frame
new_dataset.info()

new_dataset.head()

new_dataset['fraud'].value_counts()

new_dataset.groupby('fraud').mean()

#Plit into feature and target
#Drop collumns fraud
X= new_dataset.drop(columns='fraud', axis = 1) #Feature
Y= new_dataset ['fraud'] #Target

#Test data before train
print(X[:5])
print(X.shape)
print(Y[:5])
print(Y.shape)

#Plit into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

#test the shape
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

print(X.shape, X_train.shape,X_test.shape)

#Model Training
#Logistic Regression Model because binary class verification problem
model = LogisticRegression(max_iter=1000)

#Training model with training data
model.fit(X_train, Y_train)

#Model Evaluation

#Accuracy Score
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on Training data : ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score on Test Data : ', test_data_accuracy)

# Feature selection (consider feature importance analysis for best results)
features = [
    "distance_from_home",
    "distance_from_last_transaction",
    "ratio_to_median_purchase_price",
    "repeat_retailer",
    "used_chip",
    "used_pin_number",
    "online_order"
]

# Streamlit UI for user interaction
st.title("De- Tech: Check for fraudulent Credit Card transactions")
st.write("Enter transaction details for fraud prediction:")

# Create input fields for relevant features
distance_from_home = st.number_input("Distance from Home")
distance_from_last_transaction = st.number_input("Distance from Last Transaction")
ratio_to_median_purchase_price = st.number_input("Ratio to Median Purchase Price")
repeat_retailer = st.selectbox("Repeat Retailer", ("Yes", "No"))  # Assuming categorical data
used_chip = st.selectbox("Used Chip", ("Yes", "No"))  # Assuming categorical data
used_pin_number = st.selectbox("Used PIN Number", ("Yes", "No"))  # Assuming categorical data
online_order = st.selectbox("Online Order", ("Yes", "No"))  # Assuming categorical data

# Simulate data preparation (you might need additional logic in practice)
data = {
    "distance_from_home": distance_from_home,
    "distance_from_last_transaction": distance_from_last_transaction,
    "ratio_to_median_purchase_price": ratio_to_median_purchase_price,
    "repeat_retailer": repeat_retailer == "Yes",  # Convert categorical to boolean
    "used_chip": used_chip == "Yes",
    "used_pin_number": used_pin_number == "Yes",
    "online_order": online_order == "Yes"
    }

X_new = pd.DataFrame([data])

if st.button("Predict Fraud Probability"):
  fraud_probability = model.predict_proba(X_new)[0][1]  # Probability of fraud class
  fraud_label = "Fraudulent" if fraud_probability > 0.5 else "Legitimate"
  st.write(f"Predicted Fraud Probability: {fraud_probability:.2f}")
  st.write(f"Transaction Label: {fraud_label}")

