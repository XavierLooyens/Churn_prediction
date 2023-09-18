from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle

app = FastAPI()

filename = 'finalized_model.sav'
app.state.model = pickle.load(open(filename, 'rb'))

@app.get("/predict")
def predict()

    X_pred = pd.DataFrame(locals(), index=[0])

    model = app.state.model
    assert model is not None


    y_pred = model.predict(X_processed)

    return dict(fare_amount=float(y_pred))
