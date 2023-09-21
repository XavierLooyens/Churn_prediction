import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
from Churn_prediction.model import predict_churn_proba


app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.state.model = load('LogRegModel.joblib')

@app.get("/predict")
def predict():

    # Load the data
    data = ...

    # Load the model
    model = app.state.model

    #preprocess the input data
    

    #predict churn probability
    result = model.predict_proba(data)

    # Return the prediction
    return {'Churn probality': result}
