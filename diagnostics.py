"""Performing Daignostic common tests"""


import timeit
import os
import json
import pickle
import subprocess
import logging

from collections import defaultdict
import pandas as pd
from training import data_load


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Load config.json and get environment variables
with open('config.json', 'r') as f:
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

    model_path = os.path.join(deploy_folder, "trainedmodel.pkl")
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    logger.info("Extracting data for prediction")

    file_name = "testdata.csv"
    _, X_features, _ = data_load(test_data_path, file_name)

    predicted_response = model.predict(X_features)

    return predicted_response


def dataframe_summary():
    """get summary statistics

    calculate [mean,median, std] for numeric columns in the data set
    and return a list containing all summary statistics
    """

    file_name = "finaldata.csv"
    df_data, _, _ = data_load(dataset_csv_path, file_name)
    numeric_data = df_data.drop(['corporation', 'exited'], axis=1)

    logger.info("performing summary statistics")

    data_summary = numeric_data.agg(['mean', 'median', 'std'])

    return data_summary


def execution_time():
    """function execution time calcuation:

    calculate timing of training.py and ingestion.py
    returns a list of the average execution time in seconds after
    running the functions for N iterations
    """

    fnc_ref_names = ['ingestion', 'training']

    function_timings = defaultdict(list)
    iterations = 10

    logger.info(
        f"Execution time for ingestion and traing in {iterations} iteration")

    for _ in range(iterations):
        for fnc_name in fnc_ref_names:

            starttime = timeit.default_timer()
            os.system(f"python {fnc_name}.py")
            timing = timeit.default_timer() - starttime

            logger.info(f"claculating timing for: {fnc_name} function")

            function_timings[fnc_name].append(timing)

    logger.info("Calculating the average execution time")

    avg_ingest_time = sum(
        function_timings['ingestion']) / len(function_timings['ingestion'])
    avg_train_time = sum(
        function_timings['training']) / len(function_timings['training'])

    return [avg_ingest_time, avg_train_time]


def missing_data():
    """count NA in each column and thier percentage"""

    file_name = "data_train.csv"
    df_data, _, _ = data_load(dataset_csv_path, file_name)

    logger.info(f"calculating percentage of NA")

    missing_values_df = df_data.isna().sum() / df_data.shape[0]
    napercents = missing_values_df.values.tolist()

    # [Additional]: extract columns that need to be imputed
    nas = list(df_data.isna().sum())
    imp_col_idx = [ix for ix, elem in enumerate(nas) if elem != 0]

    for idx in imp_col_idx:
        df_data.iloc[:, idx].fillna(
            pd.to_numeric(df_data.iloc[:, idx], errors='coerce').mean(skipna=True),
            inplace=True)

    return str(napercents)


def outdated_packages_list():
    """get a list of outdated packated

    returns a table with three columns:
        the name of a Python module;
        the currently installed version;
        the most recent available version;
    """

    outdated_dep = subprocess.check_output(
        ['pip', 'list', '--outdated']).decode('utf-8')

    return str(outdated_dep)


if __name__ == '__main__':

    predicted_tst = model_predictions()
    summary_list = dataframe_summary()
    t_ing, t_train = execution_time()
    p_missing = missing_data()
    outdated_dep = outdated_packages_list()

    logger.info(f"Done performing necessary diagnostics ..."
                f"check related output folder\n")
