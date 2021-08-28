"""Trasfering verified output to Deployment folder"""


import os
import json
import logging

from shutil import copy2


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Load config.json and correct path variable
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
model_folder = os.path.join(config['output_model_path'])


def store_model_into_pickle():
    """copy latest files and metadata to deployment folder

    copy the latest pickle file, the latestscore.txt value,
    and the ingestfiles.txt file into the deployment directory
    """

    names_and_paths = {
        "trainedmodel.pkl": model_folder,
        "latestscore.txt": model_folder,
        'ingestedfiles.txt': dataset_csv_path
    }

    logger.info("copying files to deployment folder")

    for name, folder in names_and_paths.items():

        copy2(os.path.join(folder, name), prod_deployment_path)


if __name__ == "__main__":
    store_model_into_pickle()
    logger.info("Done copying ...check deployment folder for output\n")
