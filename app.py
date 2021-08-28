"""Flask Endpoints to train/track ML pipline"""
import json
import os
import logging
import pandas as pd

from flask import Flask, session, jsonify, request

import diagnostics 
import scoring


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_folder = os.path.join(config['output_model_path'])

prediction_model = None

logger.info("Prediction Endpoint")

@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    """ 
    take a dataset's file location as its input, 
    returns value for prediction outputs
    """
    input_file = request.args.get('file_location')
    
    folder = os.path.dirname(input_file)
    filename = os.path.basename(input_file)

    predicted = diagnostics.model_predictions(folder,filename)

    return str(predicted)

logger.info("Scoring Endpoint")

@app.route("/scoring", methods=['GET','OPTIONS'])
def score_stats():        
    """check the score of the deployed model, returns F1 score"""
    
    f1_score_ = scoring.score_model()
    return str(f1_score_)


logger.info("Summary Statistics Endpoint")

@app.route("/summarystats", methods=['GET','OPTIONS'])
def summary_stats():        
    """
    check means, medians, and std for each column
    return a list of all calculated summary statistics
    """
    
    summary_list = diagnostics.dataframe_summary()

    return str(summary_list)

logger.info("Diagnostics Endpoint")

@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diag_stats():        
    """
    returns execution time (data_ingestion, training)
    percent NA values
    and list of outdated_packages
    """
    p_missing = diagnostics.missing_data()
    ingest_time, train_time = diagnostics.execution_time()
    outdated_files = diagnostics.outdated_packages_list()

    all_output = "Missing Data:\n" +p_missing +"\nTime:\nIngestion: " + str(ingest_time) + \
                "\tTrain: "+ str(train_time) + "\nOutdated Packages:\n" + outdated_files

    return all_output
    

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
