import numpy as np
import pandas as pd



train_data=pd.read_csv('train_bank.csv')
train_data['Loan_Amount_Requested'] = train_data['Loan_Amount_Requested'].str.replace(',', '').astype(float)

cat_list=train_data.select_dtypes(include=['object']).columns
cat_list=cat_list.tolist()
cat_list

num_list=train_data.select_dtypes(include=['int64', 'float64']).columns
num_list=num_list.tolist()
num_list
#EDA
train_data.drop(['Number_Open_Accounts'], axis=1, inplace=True)
train_data['Annual_Income'].fillna((train_data['Annual_Income'].mean()), inplace=True)
train_data["Months_Since_Deliquency"] = train_data["Months_Since_Deliquency"].fillna(0)
train_data.Home_Owner.fillna("N", inplace=True)

def home_estimator(i):
    """Grouping Cabin feature by the first letter"""
    a = 0
    if i<63000:
        a = "Rent"
    elif i>=63000 and i<66000:
        a = "Other"
    elif i>=66000 and i<70000:
        a = "Own"
    elif i>=70000 and i<83000:
        a = "None"
    else:
        a = "Mortgage"
    return a

train_home_with_N = train_data[train_data.Home_Owner == "N"]
train_home_without_N = train_data[train_data.Home_Owner != "N"]
train_home_with_N.loc[:,'Home_Owner'] = train_home_with_N.Annual_Income.apply(lambda x: home_estimator(x))
train_data = pd.concat([train_home_with_N, train_home_without_N], axis=0)

def length_estimator(i):
    """Grouping Length feature by the first letter"""
    a = 0
    if i<68000:
        a = "< 1 year"
    elif i>=68000 and i<72000:
        a = "2 years"
    elif i>=72000 and i<73000:
        a = "3 years"
    elif i>=73000 and i<78000:
        a = "9 years"
    else:
        a = "10+ years"
    return a


train_data.Length_Employed.fillna("N", inplace=True)
train_length_with_N = train_data[train_data.Length_Employed == "N"]
train_length_without_N = train_data[train_data.Length_Employed != "N"]
train_data = pd.concat([train_length_with_N, train_length_without_N], axis=0)

target_feature=train_data['Interest_Rate']
train_data.drop(['Interest_Rate','Loan_ID'], axis=1, inplace=True)

from sklearn import preprocessing
labelEncoder=preprocessing.LabelEncoder()
#creating an empty dictionary to store our labels for reference
mapping_dict={}
for col in cat_list:
    train_data[col]=labelEncoder.fit_transform(train_data[col])#applying the encoder
    #using the zip and dict functions to merge(map) the unique values to their labels
    #for gender...labelEncoder.classes_ gives male, female... applying transform on labelEncoder.transform on classes_ gives the labels
    my_map=dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
    #putting into our empty dictionary
    mapping_dict[col]=my_map

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_data, target_feature, test_size=0.2)
from lightgbm import LGBMClassifier

lgb=LGBMClassifier()

lgb.fit(X_train, y_train)
#lets pickle for later use
import pickle

pickle.dump(lgb, open("lgb.pkl", "wb"))