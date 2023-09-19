from fastapi import FastAPI,File,UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd

app = FastAPI()

filename = 'finalized_model.sav'
app.state.model = pickle.load(open(filename, 'rb'))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.post("/uploadcsv/")
def upload_csv(csv_file: UploadFile = File(...)):
    X_pred= pd.read_csv(csv_file.file)
    ids= X_pred.msno

    X_columns = X_pred.columns.to_list()
    model = app.state.model
    prediction = model.predict_proba(X_pred)*100
    prediction_df = pd.DataFrame({'id': ids, 'prediction percentage': prediction[:,1]})

    return prediction_df.to_dict()




