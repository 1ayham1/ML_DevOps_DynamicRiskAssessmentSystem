from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

#Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 


def data_load():
    """load data from destination folder and return a df"""



def data_split(df):
    """split data into train/test"""


def train_model():
    """Function for training the model"""

    #use this logistic regression for training
    logit = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='ovr', n_jobs=None, penalty='l2',
                    random_state=0, solver='newton-cg', tol=0.0001, verbose=0,
                    warm_start=False)
    
    #read data
    file_name = os.path.join(dataset_csv_path,"finaldata.csv")
    df = pd.read_csv(file_name,low_memory=False)
    
    #fit the logistic regression to your data
    
    X = df.loc[:,['lastmonth_activity','lastyear_activity','number_of_employees']].values.reshape(-1, 3)
    y = df['exited'].values.reshape(-1, 1).ravel()
    
    #Alternative extraction:
    #X = df.copy().drop(["corporation"], axis=1)
    #X = X.values.reshape(-1,len(df.columns)-2)  
    #y = X.pop("exited")
    #y = y.values.reshape(-1,1).ravel()

    model = logit.fit(X, y)

    #write the trained model to your workspace in a file called trainedmodel.pkl
    train_name = os.path.join(model_path, "trainedmodel.pkl")

    pickle.dump(model, open(train_name, 'wb'))

if __name__ == "__main__":
    train_model()    
