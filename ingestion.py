"""Multiple file data ingestion, processing and fusion"""

import os
import json
import glob
import logging

import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


def merge_multiple_dataframe():
    """merge data from multiple files

    check for datasets from multiple .csv, compile them together,
    and write to an output file. It is assumed that data is spred
    across multiple folders and subfolders. for one level search use:

        glob.glob(f'{r_path}/*.csv') 

    """

    col_names = ['corporation', 'lastmonth_activity', 'lastyear_activity',
                 'number_of_employees', 'exited']

    df_list = pd.DataFrame(columns=col_names)

    logger.info("reading and merging all found .csv files")

    # recursivly search direcotries and read .csv files.
    r_path = os.path.join(os.getcwd(), input_folder_path)

    datasets = glob.glob(f'{r_path}/**/*.csv', recursive=True)
    df_list = pd.concat(map(pd.read_csv, datasets))

    logger.info("clean data and write to outputfolder")

    final_data = df_list.drop_duplicates()
    final_data.to_csv(os.path.join(output_folder_path,
                      'finaldata.csv'), index=False)

    logger.info("extract and save consumed filenames")
    # FUTURE: consider also save the path and output to .json

    file_names = [os.path.basename(path) for path in datasets]

    with open(os.path.join(output_folder_path, 'ingestedfiles.txt'), "w") as f:
        for element in file_names:
            f.write(element + "\n")


if __name__ == '__main__':

    merge_multiple_dataframe()
    logger.info("welldone ...check relevent folders for output\n")
