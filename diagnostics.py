"""Performing Daignostic common tests"""

import pandas as pd
import timeit
import os
import json
import pickle
import subprocess
import logging

from training import data_load


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

#Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
deploy_folder = os.path.join(config['prod_deployment_path'])


def model_predictions():
    """perform prediction on deployed model

    read the deployed model and a test dataset, run the model
    and returns a list containing all predictions
    """

    logger.info("loading deployed model")

    model_path = os.path.join(deploy_folder,"trainedmodel.pkl")
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    logger.info("Extracting data for prediction")
    file_name = "testdata.csv"
    _, X_features, _ = data_load(test_data_path, file_name)

    predicted = model.predict(X_features)

    return predicted

#Function to get summary statistics
def dataframe_summary():
    """
    calculate summary statistics here
    return a list containing all summary statistics
    [mean, median, std for each numeric column]
    """
    injested_name = os.path.join(dataset_csv_path,"finaldata.csv")
    df = pd.read_csv(injested_name,low_memory=False)

    numeric_data = df.drop('exited',axis=1)
    #numeric_data = df.loc[:,['lastmonth_activity','lastyear_activity','number_of_employees']]
    
    group_mean = df.mean()
    group_std = df.std()
    group_median = df.median()

    #the required output is vage ==> it will be mean, median, std for every class
    stat_list = []

    for idx in range(len(group_mean)):
        stat_list.extend([group_mean[idx],group_median[idx],group_std[idx]])
    
    return str(stat_list)

#Function to get timings
def execution_time():
    """
    calculate timing of training.py and ingestion.py
    returns a list of 2 timing values in seconds
    """
    #turn into a decorator later

    def ingestion_timing():
        starttime = timeit.default_timer()
        os.system('python ingestion.py')
        timing=timeit.default_timer() - starttime
        return timing

    def training_timing():
        starttime = timeit.default_timer()
        os.system('python training.py')
        timing=timeit.default_timer() - starttime
        return timing
    
    ingestion_time = ingestion_timing()
    training_time = training_timing()

    return [ingestion_time, training_time]


def missing_data():
    """
    count the number of NA values in each column and thier percentage
    """
    injested_name = os.path.join(dataset_csv_path,"finaldata.csv")
    df = pd.read_csv(injested_name,low_memory=False)

    nas=list(df.isna().sum())
    napercents=[nas[i]/len(df.index) for i in range(len(nas))]

    #extract columns that need to be imputed
    imp_col_idx = [ix for ix, elem in enumerate(nas) if elem !=0]

    for idx in imp_col_idx:
        df.iloc[:,idx].fillna(pd.to_numeric(df.iloc[:,idx],errors='coerce').mean(skipna=True),
                             inplace = True)   

    return str(napercents)


#Function to check dependencies
def outdated_packages_list():
    """get a list of outdated packated
    
    returns a table with three columns: 
        the name of a Python module;
        the currently installed version;
        the most recent available version;
    """
    
    outdated = subprocess.check_output(['pip', 'list','--outdated']).decode('utf-8')
   
    return str(outdated)

if __name__ == '__main__':

    #read data
    file_name = os.path.join(test_data_path,"testdata.csv")
    df = pd.read_csv(file_name,low_memory=False)

    predicted = model_predictions(df)
    #print(predicted)
    out_list = dataframe_summary()
    p_missing = missing_data()
    t_inj, t_train = execution_time()
    outdated = outdated_packages_list()
    logger.info(f"Done performing necessary diagnostics ...",
                f"check related output folder\n")





    
