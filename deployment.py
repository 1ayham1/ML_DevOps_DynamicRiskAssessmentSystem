from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
from shutil import copy


#Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
model_folder = os.path.join(config['output_model_path'])

#function for deployment
def store_model_into_pickle(model):
    """
    copy the latest pickle file, the latestscore.txt value, 
    and the ingestfiles.txt file into the deployment directory
    """

    model_path = os.path.join(model_folder,"trainedmodel.pkl")
    copy(model_path, prod_deployment_path)

    score_path = os.path.join(model_folder,"latestscore.txt")
    copy(score_path, prod_deployment_path)

    ingested_path = os.path.join(dataset_csv_path,'ingestedfiles.txt')
    copy(ingested_path, prod_deployment_path)

if __name__ == "__main__":
    store_model_into_pickle()     
    



        
        
        

