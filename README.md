# Disaster-Response-Pipeline
## Table of Contents
1. ### Installation
2. ### Project Overview
3. ### File Description
4. ### Instructions
5. ### Screenshots

## Installation
This repository was written in HTML and Python , and requires the following Python packages: 
 pandas, numpy, re, pickle, nltk, flask, json, plotly, sklearn, sqlalchemy, sys,  warnings.

## Project Overview
This code is designed to iniate a  web app which an emergency operators could exploit during a disaster (e.g. an earthquake or Tsunami), to classify a disaster text messages into several categories which then can be transmited to the responsible entity

## File Description
* **process_data.py**: This python excutuble code takes as its input csv files containing message data and message categories (labels), and then creates a SQL database
* **train_classifier.py**: This code trains the ML model with the SQL data base
* **ETL Pipeline Preparation.ipynb**:  process_data.py development procces
* **ML Pipeline Preparation.ipynb**: train_classifier.py. development procces
* **data**: This folder contains sample messages and categories datasets in csv format.
* **app**: cointains the run.py to iniate the web app.

## Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run web app: `python run.py`

4. Go to http://0.0.0.0:3001/ Or Go to http://localhost:3001/

### Screenshots
1. Main page shows the Distribution of Message Categories.
![image](https://github.com/valygnos/Disaster-Response-Pipeline/blob/844e63b450c5c711944d0ecc42c13fcf41b7ed1f/Screenshots/graph1.png)
2. Main page shows the Overview of Training Dataset.
![image](https://github.com/valygnos/Disaster-Response-Pipeline/blob/844e63b450c5c711944d0ecc42c13fcf41b7ed1f/Screenshots/graph2.png)
3. Enter message and click 'Classify Message'.
![image](https://github.com/valygnos/Disaster-Response-Pipeline/blob/844e63b450c5c711944d0ecc42c13fcf41b7ed1f/Screenshots/search_bar.png)
4. After clicking 'Classify Message', we can see the category(ies) of which the message is classified to , highlighted in green.
![image](https://github.com/valygnos/Disaster-Response-Pipeline/blob/844e63b450c5c711944d0ecc42c13fcf41b7ed1f/Screenshots/result.png)
