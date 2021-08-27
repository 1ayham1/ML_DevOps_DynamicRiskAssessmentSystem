import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

from diagnostics import model_predictions
from sklearn.metrics import confusion_matrix



#Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
model_folder = os.path.join(config['output_model_path'])

#Function for reporting
def score_model():
    """
    calculate a confusion matrix using the test data and the deployed model
    write the confusion matrix to the workspace
    """

    file_name = os.path.join(test_data_path,"testdata.csv")
    report_df = pd.read_csv(file_name,low_memory=False)

    predicted = model_predictions(report_df)
    actual_values =  report_df['exited'].values.reshape(-1, 1).ravel()

    model_cm = confusion_matrix(actual_values, predicted)
    
    #https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
    sns.set(font_scale=1.4)
    sns.heatmap(model_cm, annot=True, annot_kws={"size": 16})
    #plt.show()
    save_figure_name = os.path.join(model_folder,"confusionmatrix2.png")
    plt.savefig(save_figure_name)


if __name__ == '__main__':
    score_model()
