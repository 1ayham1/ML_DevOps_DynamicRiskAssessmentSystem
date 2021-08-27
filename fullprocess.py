import training
import scoring
import deployment
import diagnostics
import reporting

import pandas as pd

import json
import os

#Check and read new data

#Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 


input_folder_path = config['input_folder_path']
model_folder = os.path.join(config['output_model_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path'])
output_folder_path = config['output_folder_path']

"""
read ingestedfiles.txt
If there are any files in the input_folder_path directory that are not listed 
in ingestedfiles.txt, then the script runs the code in ingestion.py
to ingest all the new data.
"""
already_ingested_files = []
ingested_files = os.path.join(prod_deployment_path,"ingestedfiles.txt")

with open(ingested_files,'r') as files:
    for file in files:
        already_ingested_files.append(file)

#get a list of all files to be processed
r_path = os.path.join(os.getcwd(),input_folder_path)
new_ingested_files = []

col_names = ['corporation','lastmonth_activity','lastyear_activity',
            'number_of_employees','exited']

df_list = pd.DataFrame(columns=col_names)
all_files_ingested = True

for roots, dirs, files in os.walk(r_path):
    for file in files:
        if file not in already_ingested_files:
            
            all_files_ingested = False
            
            new_ingested_files.append(file)

            fullpath = os.path.join(roots, file)

            df = pd.read_csv(fullpath)
            df_list = df_list.append(df)



#Deciding whether to proceed, part 1
if(all_files_ingested):
    print("all files are already ingested, no need to train a new model")
    exit(0)

else:
    result = df_list.drop_duplicates()
    
    #you can of course decide to merge the old consume files and df
    data_file = os.path.join(output_folder_path,'finaldata.csv')
    result.to_csv(data_file, index=False)
    
    names_file = os.path.join(output_folder_path,'ingestedfiles.txt')
    with open(names_file,"w")  as f:
        for element in new_ingested_files:
            f.write(element+ "\n")


"""
Checking for model drift
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
    print("no model drift detected with new data set")
    print(f"new_score: {new_score}, old_score: {prev_score}")
    exit(0)
else:
    #retrun the model
    training.train_model()

#Re-deployment re-run the deployment.py script
deployment.store_model_into_pickle()

#Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model

diagnostics.model_predictions(new_ingested_files)
diagnostics.dataframe_summary()
diagnostics.execution_time()
diagnostics.missing_data()
diagnostics.outdated_packages_list()
reporting.score_model()




