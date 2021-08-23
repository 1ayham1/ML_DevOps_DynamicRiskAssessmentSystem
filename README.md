# ML_DevOps_DynamicRiskAssessmentSystem

## Objective

- To create, deploy, and monitor a risk assessment ML model that will estimate the attrition risk of each of the company's 10,000 clients. 
- If the deployed model is accurate, it will enable the client managers to contact the clients with the highest risk and avoid losing clients and revenue.
- As the industry is dynamic and constantly changing regular monitoring of the created model is setup to ensure that it remains accurate and up-to-date. This includes:
---
seting up processes and scripts to `re-train`, `re-deploy`, `monitor`, and `report` on the ML model, so that an accurate -as possible- risk assessments is obtained and an overall minimization of client attrition is achieved. 

---
## Steps

**1. Data Ingestion**

- Automatically checking a database for new data that can be used for model training.
- Compiling all training data to a training dataset and saving it to persistent storage. 
- Writing metrics related to the completed data ingestion tasks to persistent storage.

**2. Training, scoring, and deploying**

- Writing scripts that train an ML model that predicts attrition risk, and scoring the model. 
- Writing the model and the scoring metrics to persistent storage.

**3. Diagnostics**

- Determining and saving summary statistics related to a dataset. 
- Timing the performance of model training and scoring scripts. 
- Checking for dependency changes and package updates.

**4. Reporting**

- Automatically generating plots and documents that report on model metrics. 
- Providing an API endpoint that can return model predictions and metrics.

**5. Process Automation**

- Creating a script and cron job that automatically run all previous steps at regular intervals.

## File Structure
```
.
│   apicalls.py         a Python script to call API endpoints
│   app.py              a Python script that contains API endpoints
│   config.json         a data file that contains names of files for configuring ML Python scripts
│   deployment.py       a Python script to deploy a trained ML model       
│   diagnostics.py      a Python script to measure model and data diagnostics
│   fullprocess.py      a script to determine whether a model needs to be re-deployed, and to call all other Python scripts when needed
│   ingestion.py        a Python script to ingest new data
│   LICENSE
│   README.md           *this file
│   reporting.py        a Python script to generate reports about model metrics
│   requirements.txt    a text file containing current versions of all the modules the scripts use
│   scoring.py          a Python script to score an ML model
│   training.py         a Python script to train an ML model
│   wsgi.py             a Python script to help with API deployment
│   
├───ingesteddata        Contain the compiled datasets after ingestion script
├───models              contain ML models that is created for production.
├───practicedata        Contains some data for practice
│       dataset1.csv
│       dataset2.csv
│       Icon_
│       
├───practicemodels        contain ML models that are created as practice
├───production_deployment contain your final, deployed models
├───sourcedata            contains data to be loaded to train your models.
│       dataset3.csv
│       dataset4.csv
│       Icon_
│       
└───testdata              contains data for testing the models.
        Icon_
        testdata.csv
```