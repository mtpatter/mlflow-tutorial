# mlflow-tutorial

![img](assets/cartoon-serve-api.png)

I walk through this tutorial and others here on GitHub and on my [Medium blog](https://maria-patterson.medium.com/).  Here is a friend link for open access to the article on Towards Data Science: [*Machine learning model serving for newbies with MLflow*](https://towardsdatascience.com/machine-learning-model-serving-for-newbies-with-mlflow-76f9f0ac3cb2?sk=3fabd570be956c5830591f9ac0fa7991).  I'll always add friend links on my GitHub tutorials for free Medium access if you don't have a paid Medium membership [(referral link)](https://maria-patterson.medium.com/membership).  

*[edit 2024 Sep: I've updated this GitHub repo significantly since publishing my Towards Data Science article in order to upgrade to mlflow 2.16. The scripts have been updated but the Jupyter notebook is legacy.]*

If you find any of this useful, I always appreciate contributions to my Saturday morning [fancy coffee fund](https://github.com/sponsors/mtpatter)!

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

You can access the mlflow registry UI on your localhost at port 8000.

Or to run the tutorial without the registry:

```
docker compose -f docker-compose-no-registry.yml up --build
```

with the model served on port 1234.

In either case, you can then make predictions as described in the relevant section below.

To run only mlflow with Docker (without using my sklearn classifier example), port forwarding to localhost:8000,
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
mlflow models serve -m clf-model -p 1234 -h 0.0.0.0 --env-manager local
```

## Serving models with mlflow (with registry)

### Start an mlflow server for UI on port 8000

This uses a sqlite database backend and stores model artifacts
at the local specified location.

```
mlflow server \
--backend-store-uri sqlite:///mlflow.db \
--default-artifact-root ./mlflow-artifact-root \
--host 0.0.0.0 \
--port 8000
```

### Train a model

The `clf-train-registry.py` script uses the sklearn breast cancer dataset, trains a
simple random forest classifier, overwrites the model predict method to return
probabilities instead of classes, and saves and registers the model with mlflow.
The newest model is moved to the mlflow `Staging` alias.
Adding the optional flag for writing output test data will split
the training data first to add an example test data file.

```
python clf-train-registry.py clf-model "http://localhost:8000" --outputTestData test.csv
```

The model is now logged in the mlflow registry and visible in the UI under "my-experiment".

### Serve model to port 1234

Serve your latest `Staging` version of the trained `clf-model` to port 1234.

```
export MLFLOW_TRACKING_URI=http://localhost:8000
mlflow models serve -m models:/clf-model@Staging -p 1234 -h 0.0.0.0 --env-manager local
```

## Make predictions

For inference data in a file called `test.csv`, run the following:

```
curl http://localhost:1234/invocations  -H 'Content-Type: text/csv' --data-binary @test.csv
```

or just run the script below:

```
./predict.sh test.csv
```

This returns an array of predicted probabilities.

## Cleaning up

If you're using Compose, when finished, shut down all containers with the following command:

```
docker compose down
```

Note well that the compose files mount volumes and write to the local directory.
