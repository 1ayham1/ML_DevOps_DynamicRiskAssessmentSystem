from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os

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
test_data_path = os.path.join(config['test_data_path']) 

def data_load(folder_path, file_name):
    """load data from destination folder
    
    data_file    : name of the file containing required data
    folder_path  : folder where data resides
    
    return:
        df       : pandas data frame containing data
        X        : model features
        y        : model output 
    """

    file_name = os.path.join(folder_path,file_name)
    df = pd.read_csv(file_name,low_memory=False)

    logger.info("extracting features: X and target: y")
    X = df.loc[:,['lastmonth_activity','lastyear_activity','number_of_employees']].values.reshape(-1, 3)
    y = df['exited'].values.reshape(-1, 1).ravel()

    #Alternative extraction:
    #X = df.copy().drop(["corporation"], axis=1)
    #X = X.values.reshape(-1,len(df.columns)-2)  
    #y = X.pop("exited")
    #y = y.values.reshape(-1,1).ravel()

    return df, X, y


def data_split():
    """split data into train/test"""

    file_name = "finaldata.csv"
    logger.info(f"Loading {file_name} data from {dataset_csv_path}")

    df, X, y = data_load(dataset_csv_path,file_name)
    

    splits = {}

    splits["train"], splits["test"] = train_test_split(
        df,
        test_size= 0.2,
        random_state= 42,
        
    )

    for split, df in splits.items():

        # Make the artifact name from the provided root plus the name of the split
        artifact_name = f"data_{split}.csv"

        # Get the path on disk 
        save_dir = test_data_path if split=="test" else dataset_csv_path
        output_path = os.path.join(save_dir, artifact_name)

        logger.info(f"saving the {split} dataset to {artifact_name}")
        df.to_csv(output_path,index=False)




def train_model():
    """Function for training the model"""

    data_split()

    folder_path = dataset_csv_path 
    file_name = "data_train.csv"
    _, X, y = data_load(folder_path, file_name)

    #use this logistic regression for training
    #FUTURE: pass parameters from .json
    logit = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='ovr', n_jobs=None, penalty='l2',
                    random_state=0, solver='newton-cg', tol=0.0001, verbose=0,
                    warm_start=False)

    #fit the logistic regression to your data
    logger.info(f"fitting LR from: {file_name} dataset")
    model = logit.fit(X, y)

    #write the trained model to  workspace in a file called trainedmodel.pkl
    logger.info(f"saving trained model to: {model_path}")

    train_name = os.path.join(model_path, "trainedmodel.pkl")
    pickle.dump(model, open(train_name, 'wb'))

if __name__ == "__main__":
    train_model()  
    logger.info("welldone ...check relevent folders for output\n")  
