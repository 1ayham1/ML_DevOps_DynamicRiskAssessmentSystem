

"""Scoring deployed model"""

import json
import os
import logging

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from diagnostics import model_predictions
from training import data_load


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
model_folder = os.path.join(config['output_model_path'])


def score_model(ref_num=''):
    """score the model

    calculate a confusion matrix using the test data and
    the deployed model.
    write the confusion matrix to the workspace
    """

    file_name = "data_test.csv"  # "testdata.csv"
    _, _, actual_values = data_load(test_data_path, file_name)

    logger.info("scoring a deployed model")

    predicted_responses = model_predictions(test_data_path, file_name)

    model_cm = confusion_matrix(actual_values, predicted_responses)

    logger.info("saving confusion matrix plot")
    # https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
    sns.set(font_scale=1.4)
    sns.heatmap(model_cm, annot=True, annot_kws={"size": 16})

    save_figure_name = os.path.join(
        model_folder, f"confusionmatrix{ref_num}.png")
    plt.savefig(save_figure_name)


if __name__ == '__main__':
    score_model()
