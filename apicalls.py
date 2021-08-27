import requests
import os
import json

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1/"

#Call each API endpoint and store the responses
call_1 = requests.post('http://127.0.0.1:8000/prediction?file_location=testdata/testdata.csv').text
call_2 = requests.get('http://127.0.0.1:8000/scoring').text 
call_3 = requests.get('http://127.0.0.1:8000/summarystats').text
call_4 = requests.get('http://127.0.0.1:8000/diagnostics').text 

#combine all API responses
responses = call_1 + "\n" + call_2 + "\n" + call_3 + "\n" + call_4

#write the responses to your workspace

#Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

model_folder = os.path.join(config['output_model_path'])
output_file = os.path.join(model_folder,"apireturns.txt")

with open(output_file, 'w') as f:
   f.write(responses)

