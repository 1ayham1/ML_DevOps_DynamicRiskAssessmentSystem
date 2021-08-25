import pandas as pd
import numpy as np
import os
import json
from datetime import datetime


#Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 


input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

#Function for data ingestion
def merge_multiple_dataframe():
    """
    check for datasets from multiple .csv, compile them together,
     and write to an output file
    """
    col_names = ['corporation','lastmonth_activity','lastyear_activity',
                'number_of_employees','exited']
    
    df_list = pd.DataFrame(columns=col_names)
    r_path = os.path.join(os.getcwd(),input_folder_path)

    for roots, dirs, files in os.walk(r_path):
        
        for file in files:
            
            fullpath = os.path.join(roots, file)

            df = pd.read_csv(fullpath)
            df_list = df_list.append(df)

    result = df_list.drop_duplicates()
    result.to_csv('Z_result.csv', index=False)

if __name__ == '__main__':
    merge_multiple_dataframe()
