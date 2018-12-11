# XGBoost Classifier REST Interface Demo

A binary classifier demo with a REST interface.

The method is explained in [the notebook](notebooks/explore.ipynb).

## How to run

Create the conda env:

```
conda env create -f environment.yml
```

The training and the REST interface can be run after activating this environment.

```
python train.py && python serve.py
``` 

## Docker

It creates the trained model during the build.

```
docker build -t classifier-rest-demo .
```

The REST interface can be started:

```
docker run -p 8080:8080 classifier-rest-demo
```

## Interface

The web server has a single endpoint which takes a sample as JSON

```
curl --header "Content-Type: application/json"  \
    --request POST   \
    --data '{"v1": "b", "v2": 59.5, "v3": 0.000275, "v4": "u", "v5": "g", "v6": "W", "v7": "v", "v8": 1.75, "v9": "t", "v10": "t", "v11": 5, "v12": "t", "v13": "g", "v14": 60.0, "v15": 58, "v17": 600000.0, "v18": NaN, "v19": 0}' \
    http://localhost:8080/predict
```

it returns the predicted class and the probabilities (for 0,1 respectively):

```
{
  "predicted_class": 1,
  "probabilities": [
    0.06513875722885132,
    0.9348612427711487
  ]
}
```
