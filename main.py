# Put the code for your API here.
import os
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from starter.ml.data import process_data
from starter.ml.model import inference


file_dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(file_dir, 'model/model.pkl')
encoder_path = os.path.join(file_dir, 'model/encoder.pkl')
lb_path = os.path.join(file_dir, 'model/label_binarizer.pkl')

model = pickle.load(open(model_path, 'rb'))
encoder = pickle.load(open(encoder_path, 'rb'))
lb = pickle.load(open(lb_path, 'rb'))

app = FastAPI()

class CensusData(BaseModel):
    age: int = Field(None, example=39)
    workclass: str = Field(None, example='State-gov')
    fnlgt: int = Field(None, example=77516)
    education: str = Field(None, example='Bachelors')
    education_num: int = Field(None, example=13)
    marital_status: str = Field(None, example='Never-married')
    occupation: str = Field(None, example='Adm-clerical')
    relationship: str = Field(None, example='Not-in-family')
    race: str = Field(None, example='White')
    sex: str = Field(None, example='Female')
    capital_gain: int = Field(None, example=2174)
    capital_loss: int = Field(None, example=0)
    hours_per_week: int = Field(None, example=40)
    native_country: str = Field(None, example='United-States')


# GET on the root giving a welcome message.
@app.get("/")
async def welcome():
  return "Hello World!"

# POST that does model inference.
@app.post("/inference")
async def predict(dataRow: CensusData):
    dataRow = {key.replace('_', '-'): [value] for key, value in dataRow.__dict__.items()}
    data = pd.DataFrame.from_dict(dataRow)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X, _, _, _ = process_data(data, categorical_features=cat_features,
                              label=None,
                              training=False,
                              encoder=encoder, lb=lb
                            )
    # Do inference
    prediction = inference(model, X)[0]
    
    # Get prediction
    if prediction == 0: 
       pred_api = "<=50K"
    else:
       pred_api = ">50K"

    return pred_api