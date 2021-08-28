"""Process Automation"""

import json
import os
import logging
import glob

import pandas as pd

import training
import scoring
import deployment
import diagnostics
import reporting
import ingestion


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()



#Load config.json and get input and output paths
with open('config.json','r') as in_file:
    config = json.load(in_file) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
model_folder = os.path.join(config['output_model_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path'])


"""
read [ingestedfiles.txt] from where the model is deployed
If there are any files in the [input_folder_path] directory 
that are not listed in [ingestedfiles.txt], then the script
runs the code in ingestion.py to ingest all the new data.
"""

already_ingested_files = []
ingested_files = os.path.join(prod_deployment_path,"ingestedfiles.txt")

logger.info("Checking and reading already ingested files")

with open(ingested_files,'r') as files:
    for file in files:
        already_ingested_files.append(file.strip())

#check files in current director if new files are found, the procede to process them
r_path = os.path.join(os.getcwd(),input_folder_path)
current_data_files = glob.glob(f'{r_path}/**/*.csv', recursive=True)

#keep only file names
current_file_names = [os.path.basename(file) for file in current_data_files]

#subtract longer list from smaller one, so no elements are ignored
if(len(current_file_names)>=len(already_ingested_files)):
    missing_files = list(set(current_file_names).difference(already_ingested_files))
else:
    missing_files = list(set(already_ingested_files).difference(current_file_names))

if missing_files:
    
    logger.info(f"the following new files seems to be added:\n{missing_files}")
    logger.info("Checking and reading new data from input_folder")
    
    #content will be written to output_folder as ingestedfiles.txt
    ingestion.merge_multiple_dataframe()

else:
    logger.info("\n\nNo new files were detected. Exiting...")
    exit(0)


logger.info("Checking for model drift")

"""
check whether the score from the deployed model is different from the
score from the model that uses the newest ingested data
"""

#read old score from deployment direcotry
with open(os.path.join(prod_deployment_path, "latestscore.txt"), "r") as f:
    prev_score = float(f.read())

#make new prediction using stored model and new ingested data
new_score = scoring.score_model(new_data=True)

#check model drift
if new_score >= prev_score:
    logger.info("no model drift detected with new data set")
    logger.info(f"new_score: {new_score}, old_score: {prev_score}")
    
    exit(0)
else:
    logger.info(f"new_score: {new_score}, old_score: {prev_score}")
    logger.info("WARNING: ......Model Drift is detected.")
    logger.info("WARNING: ......Retraining:")

    training.train_model()


logger.info("Re-deployment re-run the deployment.py script")

deployment.store_model_into_pickle()


logger.info("Diagnostics and reporting on the re-deployed model")

#new data is already split using ingestion.py and outputed to output_folder
#This Data flow model is leaky, and will be enhanced in the future

folder = 'output_folder_path'
file_name = 'data_train.csv'

diagnostics.model_predictions(folder,file_name)
diagnostics.dataframe_summary()
diagnostics.execution_time()
diagnostics.missing_data()
diagnostics.outdated_packages_list()

"""
add a refernce number to the saved file. so it does not override previous 
plot. Equivelent to defining class static variable incremented everytime the 
function is called
"""
reporting.score_model(ref_num='2')




