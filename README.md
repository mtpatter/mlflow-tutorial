# mlflow-tutorial

This GitHub repo walks through an example of training a classifier model
with sklearn and serving the model with mlflow.
The first section saves the mlflow model locally to disk, and the second
section shows how to use the mlflow registry for model tracking and versioning.

## TLDR

To skip through and run all components with Docker Compose you can run
this whole tutorial with the registry:

```
docker compose -f docker-compose.yml up --build
```

or without the registry:

```
docker compose -f docker-compose-no-registry.yml up --build
```

with the model served on port 1234.

To run only the mlflow server with Docker, port forwarding to localhost:5000,
you can use a compose file with the command below:

```
docker compose -f compose-server.yml up --build
```

## Serving models with mlflow (no registry)

### Train a model

The `clf-train.py` script uses the sklearn breast cancer dataset, trains a
simple random forest classifier, and saves the model to local disk with mlflow.
Adding the optional flag for writing output test data will split
the training data first to add an example test data file.

```
python clf-train.py clf-model --outputTestData test.csv
```

### Serve model to port 1234

Serve your trained `clf-model` to port 1234.

```
mlflow models serve -m clf-model -p 1234 -h 0.0.0.0
```

## Serving models with mlflow (with registry)

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

The `clf-train-registry.py` script uses the sklearn breast cancer dataset, trains a
simple random forest classifier, overwrites the model predict method to return
probabilities instead of classes, and saves and registers the model with mlflow.
The newest model is moved to the mlflow `Staging` version.
Edit for your own models or preferred stage or versions.
Adding the optional flag for writing output test data will split
the training data first to add an example test data file.

```
python clf-train-registry.py clf-model "http://localhost:5000" --outputTestData test.csv
```

The jupyter notebook also works through model training and serving with mlflow.

### Serve model to port 1234

Serve your latest `Staging` version of the trained `clf-model` to port 1234.

```
export MLFLOW_TRACKING_URI=http://localhost:5000
mlflow models serve -m models:/clf-model/Staging -p 1234 -h 0.0.0.0
```

## Make predictions

For inference data in a file called `test.csv`, run the following:

```
curl http://localhost:1234/invocations  -H 'Content-Type: text/csv' --data-binary @test.csv
```

or just run the script (after making executable):

```
predict.sh test.csv
```

This returns an array of predicted probabilities.

## Cleaning up

If you're using Compose, when finished, shut down all containers with the following command:

```
docker compose down
```

Note well that the compose files mount volumes and write to the local directory.
