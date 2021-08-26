from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

#Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
model_folder = os.path.join(config['output_model_path']) 

#Function for model scoring
def score_model():
    """
    this function takes a trained model, 
    load test data, and calculate an F1 score for the model relative 
    to the test data. it writes the result to the latestscore.txt file
    """

    #load the model
    model_path = os.path.join(model_folder,"trainedmodel.pkl")
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    #read data
    file_name = os.path.join(test_data_path,"testdata.csv")
    df = pd.read_csv(file_name,low_memory=False)

    X = df.loc[:,['lastmonth_activity','lastyear_activity','number_of_employees']].values.reshape(-1, 3)
    y = df['exited'].values.reshape(-1, 1).ravel()

    predicted = model.predict(X)

    f1score = metrics.f1_score(predicted,y)
    
    score_path = os.path.join(model_folder,"latestscore.txt")

    with open(score_path, 'w') as f:
        f.write(str(f1score))

if __name__ == "__main__":
    score_model()     
