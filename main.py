from fastapi import FastAPI
import joblib
from pydantic import BaseModel, Field
import pandas as pd

# load scaler and trained model
scaler = joblib.load("models/scaler.pkl")
model = joblib.load("models/model.pkl")

# create FastAPI app
app = FastAPI()

# input data model
class BreastCancerData(BaseModel):
    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    concave_points_mean: float = Field(alias="concave points_mean")
    symmetry_mean: float
    fractal_dimension_mean: float

    radius_se: float
    texture_se: float
    perimeter_se: float
    area_se: float
    smoothness_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float = Field(alias="concave points_se")
    symmetry_se: float
    fractal_dimension_se: float

    radius_worst: float
    texture_worst: float
    perimeter_worst: float
    area_worst: float
    smoothness_worst: float
    compactness_worst: float
    concavity_worst: float
    concave_points_worst: float = Field(alias="concave points_worst")
    symmetry_worst: float
    fractal_dimension_worst: float

# home endpoint
@app.get("/")
def home():
    return {"message": "Breast Cancer Prediction API"}

# prediction endpoint
@app.post("/predict")
def predict(data: BreastCancerData):

    new_data = pd.DataFrame([data.model_dump(by_alias=True)])

    new_data_scaled = scaler.transform(new_data)

    prediction = model.predict(new_data_scaled)

    if prediction[0] == 0:
        result = "Benign"
    else:
        result = "Malignant"

    return {"prediction": result}