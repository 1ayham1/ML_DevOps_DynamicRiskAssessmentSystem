"""Calculating F1-Score of a trained model"""

import pickle
import os
import json
import logging

from sklearn import metrics
from training import data_load


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
model_folder = os.path.join(config['prod_deployment_path'])


def score_model(new_data=False):
    """Function for model scoring

    Given a trained model, relevent test data is loaded, and F1 score is
    then calculated. Result is written back to appropriate folder with
    the name [latestscore.txt]

    new_data : if new data is available, calculate the score on
                related training data
    """

    logger.info("read and load trained model")

    model_path = os.path.join(model_folder, "trainedmodel.pkl")
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # read data, if new data is ingested, then new data is used for scoring
    if new_data:

        logger.info("Scoring data_train: new data is ingested.")

        file_name = "data_train.csv"
        _, X_features, y_response = data_load(dataset_csv_path, file_name)

    else:
        logger.info("Scoring data_test")

        file_name = "data_test.csv"
        _, X_features, y_response = data_load(test_data_path, file_name)

    logger.info("Model prediction and calculating F1 score")

    predicted = model.predict(X_features)
    model_score = metrics.f1_score(predicted, y_response)

    logger.info("Saving F1 score to a file")

    score_path = os.path.join(model_folder, "latestscore.txt")
    with open(score_path, 'w') as file_write:
        file_write.write(str(model_score))

    return model_score


if __name__ == "__main__":
    out_score = score_model()
    logger.info(f"\nwelldone f1_score is: {out_score}...\n"
                f"check relevent folders for output")
