import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load

app = FastAPI()

# load model
model = load('lr.joblib')

# define types of inputs
class WineComposition(BaseModel):
    alcohol: float
    chlorides: float
    citric_acid: float
    density: float
    fixed_acidity: float
    free_sulfur_dioxide: int
    pH: float
    residual_sugar: float
    sulphates: float
    total_sulfur_dioxide: int
    volatile_acidity: int

# define prediction endpoint
@app.post("/predictions")
async def make_prediction(wine_composition: WineComposition):
    # transform input into dataframe
    pandas_input = pd.DataFrame.from_dict([wine_composition.dict()])
    # predict
    result = model.predict(pandas_input)
    # transform result into JSON and return it to user
    return {'wine_quality': result.tolist()[0]}