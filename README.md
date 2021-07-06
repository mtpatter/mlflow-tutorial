# mlflow-tutorial

## Serving models with mlflow

### Start an mlflow server for UI on port 5000

This uses a sqlite database backend and stores model artifacts
at the local specified location.

```
mlflow server \
--backend-store-uri sqlite:///mlflow.db \
--default-artifact-root ./mlflow-artifact-root \
--host 0.0.0.0
```

### Train a model

The `clf-train.py` script uses the sklearn breast cancer dataset, trains a
simple random forest classifier, overwrites the model predict method to return
probabilities instead of classes, and saves and registers the model with mlflow.
The newest model is moved to the mlflow `Staging` version.
Edit for your own models or preferred stage or versions.
Adding the optional flag for writing output test data will split
the training data first to add an example test data file.

```
python clf-train.py clf-model "http://localhost:5000" --outputTestData test.csv
```

The jupyter notebook also works through model training and serving with mlflow.

### Serve model to port 1234

Serve your latest `Staging` version of the trained `clf-model` to port 1234.

```
export MLFLOW_TRACKING_URI=http://localhost:5000
mlflow models serve -m models:/clf-model/Staging -p 1234 -h 0.0.0.0
```

### Make predictions

For inference data in a file called `test.csv`, run the following:

```
curl http://localhost:1234/invocations  -H 'Content-Type: text/csv' --data-binary @test.csv
```

This returns an array of predicted probabilities.


## Using Docker

To run only the mlflow server with Docker, port forwarding to localhost:5000,
you can use a compose file with the command below:

```
docker compose -f compose-server.yml up --build
```

To run this whole tutorial, starting the server, training a model, and serving
it to port 1234, use the main compose file with the command below:

```
docker compose -f docker-compose.yml up --build
```

You can then still access the served model locally at port 1234 via a curl command:

```
curl http://localhost:1234/invocations  -H 'Content-Type: text/csv' --data-binary @test.csv
```

When finished, shut down all containers with the following command:

```
docker compose down
```

Note well that the compose files mount volumes and write to the local directory.
