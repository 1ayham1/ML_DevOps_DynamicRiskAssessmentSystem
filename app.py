from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle

import json
import os

from diagnostics import model_predictions,dataframe_summary,execution_time,missing_data,outdated_packages_list
from scoring import score_model

#Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_folder = os.path.join(config['output_model_path'])

prediction_model = None

#Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    """ 
    take a dataset's file location as its input, 
    returns value for prediction outputs
    """
    #file_location = request.args.get('file_location')
    #filename = os.path.join(file_location,"testdata.csv")

    #or pass file name directly using fullpath .../..../x.csv
    filename = request.args.get('file_location')
    
    data_df = pd.read_csv(filename)
    predicted = model_predictions(data_df)

    return str(predicted)


#Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score_stats():        
    """check the score of the deployed model, returns F1 score"""
    
    f1score = score_model()
    return str(f1score)


#Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summary_stats():        
    """
    check means, medians, and std for each column
    return a list of all calculated summary statistics
    """
    
    summary_list = dataframe_summary()

    return str(summary_list)


#Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diag_stats():        
    """
    returns execution time (data_ingestion, training)
    percent NA values
    and list of outdated_packages
    """
    p_missing = missing_data()
    ingest_time, train_time = execution_time()
    outdated = outdated_packages_list()

    all_output = "Missing Data:\n" +p_missing +"\nTime:\nIngestion: " + str(ingest_time) + \
                "\tTrain: "+ str(train_time) + "\nOutdated Packages:\n" + outdated

    return all_output
    



if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
