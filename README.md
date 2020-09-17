# ml-kube-deployment

This project shows you how to: 

* Train a Machine Learning model with sklearn

* Serve the model behind an API with Docker 

* Deploy the docker image to Kubernetes on GCP

## Use case

Here, we will train a ML model in order to predict the wine quality based on chimical properties. The model will then give a score between 0 and 5. I will then use this model and serve requests behind an API with FastAPI.

### Train

```python
import warnings
import sys

import pandas as pd
import numpy as np

import logging

from joblib import dump, load
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

np.random.seed(40)

# Read the wine-quality csv file from the URL
csv_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

try:
    data = pd.read_csv(csv_url, sep=";")
except Exception as e:
    logger.exception(
        "Unable to download training & test CSV, check your internet connection. Error: %s",
        e,
    )

# Split the data into training and test sets. (0.75, 0.25) split.
train, test = train_test_split(data)

# The predicted column is "quality" which is a scalar from [3, 9]
train_x = train.drop(["quality"], axis=1)
test_x = test.drop(["quality"], axis=1)
train_y = train[["quality"]]
test_y = test[["quality"]]

alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5


lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
lr.fit(train_x, train_y)

predicted_qualities = lr.predict(test_x)

(rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

dump(lr, 'lr.joblib')
```

### Serving

```python
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
```

## Train the model

You will need `virtualenwrapper` to create a virtual environment

```bash
mkvirtualenv ml-kube-deployment
pip install -r requirements.txt
python train.py
```

## Build docker container and serve model behind an API on your machine

```bash
docker build --tag=test .
docker run -p 8000:8000 test
```

You should now be able to test your API through the Swagger interface on [](http://localhost:8000/docs)

## Deploy container on Kubernetes (GCP)

### Setup

First and foremost, you will need a GCP account and you will need to create a Project on GCP.

When this is done, configure the Google Cloud SDK : [](https://cloud.google.com/sdk/docs/install).


To ease the next steps, please specify the GCP PROJECT_ID into an environment variable like this : 

```bash
export GCP_PROJECT_ID=REPLACE_HERE
```

You are now ready to go !

### Build container and push it to GCR

First, let's configure Docker to speak to the Container Registry.

```bash
gcloud auth configure-docker
```

Then build the docker image :

```bash
docker build -t gcr.io/$GCP_PROJECT_ID/ml-kube-deployment .
```

And push it to the Container Registry !

```bash
docker push gcr.io/$GCP_PROJECT_ID/ml-kube-deployment
```

### Deploy to Kubernetes

First let's create the Kubernetes cluster (this may take a while).

```bash
gcloud container clusters create ml-cluster --num-nodes=2
```

Let's deploy the application in Kubernetes : 


```bash
kubectl create deployment ml-kube-deployment --image=gcr.io/${GCP_PROJECT_ID}/ml-kube-deployment
```

And now, we need to make this application available to the world ! 

Note that here we specify the target port which is the port on which we will redirect the API requests to the container ! This may take a wh

```bash
kubectl expose deployment ml-kube-deployment --type=LoadBalancer --port 80 --target-port 8000

```

Then to get the deployed IP, type this command, and wait for the Load Balancer to get an External IP (this can take up to 2 minutes).


```bash
kubectl get svc --watch
```

For example this is what I got : 

```
(ml-kube-deployment) DMFR0900:ml-kube-deployment p.genissel$ kubectl get svc --watch
NAME                 TYPE           CLUSTER-IP     EXTERNAL-IP   PORT(S)        AGE
kubernetes           ClusterIP      10.3.240.1     <none>        443/TCP        4m5s
ml-kube-deployment   LoadBalancer   10.3.246.229   <pending>     80:30661/TCP   11s
ml-kube-deployment   LoadBalancer   10.3.246.229   34.74.166.9   80:30661/TCP   38s
```

Then I was able to interact with my API on [](http://34.74.166.9)

### Cleanup

Don't forget to destroy your Kubernetes cluster when you're done testing ! 

First, let's delete the application on the cluster.

```bash
kubectl delete deployment ml-kube-deployment
```

And now destroy the Kubernetes cluster:

```bash
gcloud container clusters delete ml-cluster
```
