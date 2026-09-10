# Breast Cancer Prediction Microservice

A repository featuring a Breast Cancer Prediction microservice and uses FastAPI.

The microservice project uses the Breast Cancer Wisconsin dataset, which consists of 569 values and 30 columns.
Each record also consists of:
- diagnosis (either Benign or Malignant)
- radius
- texture
- perimeter
- area
- smoothness
- compactness
- concavity
- concave points
- symmetry

and fractal_dimension mean values.

The main column variables (except diagnosis column) are provided in standard error and worst values each.

## Packages

For the notebook:
- pandas
- numpy
- sklearn
    - train_test_split
    - StandardScaler
    - LogisticRegression
    - accuracy_score
    - classification_report
- joblib

For the app:
- fastapi
- joblib
- pydantic
    - BaseModel
    - Field
- pandas

## Structure

    breast-cancer-prediction-microservice/
    ├── data/
    │   ├── data_upd.csv
    │
    ├── models/
    │   ├──model.pkl
    │   ├──scaler.pkl
    │   
    ├── notebook/
    │   ├── Notebook1.ipynb
    │
    ├── main.py
    ├── requirements.txt
    └── README.md

## Environment Setup
### Create a Virtual Environment

Inside the terminal, execute the following command:

    python -m venv venv

Then:

    venv\Scripts\activate

If you want to deactivate your virtual environment:

    deactivate

## Notebook
### Import Libraries
First, import necessary libraries/packages needed for the process to take place.

### Read Dataframe from CSV File

It is important to read the dataset and create a dataframe before preprocessing.
Note that the dataset needs to be downloaded locally in order to be used.

- If stored in a sub-folder, you need to paste the dataset directory into the notebook.
- If stored in the project folder, you need to use the dataset's name instead of pasting the directory.

### Drop Prediction Column/Target Value

To predict the final result, it is important to drop the target value column.
In this case, it is the `diagnosis` column, which represents the model's target value.

### Train-Test Split and Data Scaling

Before data preprocessing/scaling and/or model training processes, the data is split into train and test set.
In this case, it is important the test size be set to 0.2 so that a result of 80% is validated during training.

Additionally, the data is scaled such that they should fit the classifier used in this model.

### Data Classification and Prediction

In this case, data classification is performed using Logistic Regression, since data have been scaled for this purpose.
Before evaluating the model, the target value is predicted.

### Data Evaluation

The data is evaluated using accuracy and classification report.

### Save Model and Scaler

After training and evaluating the data, save the model and scaler as two individual files.
The model and scaler are in `.pkl` format for future deployment.

In this case, the scaler is optional but important for API testing purposes during app development.

## Application
### Load Scaler and Trained Model

First, load the scaler and the trained model from the folder they are stored.
Due to the application development structure, it is not necessary to paste the directory of both scaler and model.

### Create FastAPI app
Create a FastAPI app and test its APIs by typing the following command into your terminal:

    uvicorn main:app --reload

Then, copy your local server link into your browser to test if API creation is fully functional.
To view APIs, type `/docs`.

### Set Input Values

To create an input data model, paste the dataset's columns into your code and set type values for the prediction to work.
In this case, all columns are set to float.

### Create Endpoints

To satisfy API features, it is important endpoints be created for each feature.
In the current example, the application consists of two endpoints, GET and POST.

The GET endpoint is used to retrieve data from the server, while the POST endpoint sends data to the server.
