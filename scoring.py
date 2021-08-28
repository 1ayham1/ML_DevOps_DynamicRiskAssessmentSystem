import pickle
import os
import json

import pandas as pd
from sklearn import metrics




#Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
#model_folder = os.path.join(config['output_model_path']) 
model_folder = os.path.join(config['prod_deployment_path']) 

#Function for model scoring
def score_model(new_data = False):
    """
    this function takes a trained model, 
    load test data, and calculate an F1 score for the model relative 
    to the test data. it writes the result to the latestscore.txt file
    """

    #load the model
    model_path = os.path.join(model_folder,"trainedmodel.pkl")
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    #read data, if we are ingesting new data, then new data is used for scoring
    if(new_data):
        
        file_name = os.path.join(dataset_csv_path,"finaldata.csv")
    
    else:
        file_name = os.path.join(test_data_path,"testdata.csv")
    
    df = pd.read_csv(file_name,low_memory=False)

    X = df.loc[:,['lastmonth_activity','lastyear_activity','number_of_employees']].values.reshape(-1, 3)
    y = df['exited'].values.reshape(-1, 1).ravel()

    predicted = model.predict(X)

    f1score = metrics.f1_score(predicted,y)

    #writing score
    score_path = os.path.join(model_folder,"latestscore.txt")
    with open(score_path, 'w') as f:
        f.write(str(f1score))
    
    return f1score




