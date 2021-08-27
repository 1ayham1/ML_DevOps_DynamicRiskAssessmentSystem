from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import create_prediction_model
import diagnosis 
import predict_exited_from_saved_model
import json
import os


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
    filename = request.args.get('file')
    data = pd.read_csv(filename)

    model_path = os.path.join(model_folder,"trainedmodel.pkl")
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    prediction= model.predict(data)

    return str(prediction)

#Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def stats():        
    #check the score of the deployed model
    #add return value (a single F1 score number)
    return None

#Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    #check means, medians, and modes for each column
    #return a list of all calculated summary statistics
    return None

#Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def stats():        
    #check timing and percent NA values
    #add return value for all diagnostics
    return None

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
