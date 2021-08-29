"""Calling API Endpoints and reporting"""
import requests
import os
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Specify a URL that resolves to workspace
URL = "http://127.0.0.1/"

logger.info("Calling each API endpoint and store the responses")

prediction_call = requests.post(
    'http://127.0.0.1:8000/prediction?file_location=testdata/data_test.csv').text
scoring_call = requests.get('http://127.0.0.1:8000/scoring').text
stat_call = requests.get('http://127.0.0.1:8000/summarystats').text
diag_call = requests.get('http://127.0.0.1:8000/diagnostics').text

# combine all API responses
responses = (
    f"Perdiction Output:\n {prediction_call}\n"
    f"Score:\n  {scoring_call}\n"
    f"Summary Stat:\n {stat_call}\n"
    f"Diagnostic:\n {diag_call}")
                
                
                


# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

model_folder = os.path.join(config['output_model_path'])
output_file = os.path.join(model_folder, "apireturns.txt")

# [FIX Later] just to account for writing another file in case of model drift.
if(os.path.isfile(output_file)):
    output_file = os.path.join(model_folder, "apireturns2.txt")

logger.info("write the responses to workspace")

with open(output_file, 'w') as f:
    f.write(responses)
