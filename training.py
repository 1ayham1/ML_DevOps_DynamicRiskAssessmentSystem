"""Data Split and Training"""


import pickle
import os
import json
import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])
test_data_path = os.path.join(config['test_data_path'])


def data_load(folder_path, file_name):
    """loads data from destination folder

    data_file    : name of the file containing required data
    folder_path  : folder where data resides

    return:
        df_data           : pandas data frame containing data
        y_response        : model features
        y_response        : model output
    """

    file_name = os.path.join(folder_path, file_name)
    df_data = pd.read_csv(file_name, low_memory=False)

    logger.info("extracting features: X and target: y")

    X_features = df_data.copy().drop(["corporation"], axis=1)
    y_response = X_features.pop("exited")

    X_features = X_features.values.reshape(-1, len(df_data.columns) - 2)
    y_response = y_response.values.reshape(-1, 1).ravel()

    # Alternative extraction:
    #X = df.loc[:,['lastmonth_activity','lastyear_activity','number_of_employees']].values.reshape(-1, 3)
    #y = df['exited'].values.reshape(-1, 1).ravel()

    return df_data, X_features, y_response


def data_split():
    """split data into train/test"""

    file_name = "finaldata.csv"
    logger.info(f"Loading {file_name} data from {dataset_csv_path}")

    df_data, _, _ = data_load(dataset_csv_path, file_name)

    splits = {}

    splits["train"], splits["test"] = train_test_split(
        df_data,
        test_size=0.1,
        random_state=42,

    )

    for split, extract_data in splits.items():

        # artifact name from the provided root plus the name of the split
        artifact_name = f"data_{split}.csv"

        # Get the path on disk
        save_dir = test_data_path if split == "test" else dataset_csv_path
        output_path = os.path.join(save_dir, artifact_name)

        logger.info(f"saving the {split} dataset to {artifact_name}")
        extract_data.to_csv(output_path, index=False)


def train_model():
    """Function for training the model"""

    data_split()

    folder_path = dataset_csv_path
    file_name = "data_train.csv"
    _, X_features, y_response = data_load(folder_path, file_name)

    # use this logistic regression for training
    # FUTURE: pass parameters from .json
    logit = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class='ovr',
        n_jobs=None,
        penalty='l2',
        random_state=0,
        solver='newton-cg',
        tol=0.0001,
        verbose=0,
        warm_start=False)

    # fit the logistic regression to your data
    logger.info(f"fitting LR from: {file_name} dataset")
    model = logit.fit(X_features, y_response)

    # write the trained model to  workspace in a file called trainedmodel.pkl
    logger.info(f"saving trained model to: {model_path}")

    train_name = os.path.join(model_path, "trainedmodel.pkl")
    pickle.dump(model, open(train_name, 'wb'))


if __name__ == "__main__":
    train_model()
    logger.info("welldone ...check relevent folders for output\n")
