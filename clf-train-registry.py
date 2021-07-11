#!/usr/bin/env python

"""Example for training a random forest classifier in sklearn
   and using mlflow to save and register a model.
"""

import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
import time
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus


def wait_model_transition(model_name, model_version, stage):
    client = MlflowClient()
    for _ in range(10):
        model_version_details = client.get_model_version(name=model_name,
                                                         version=model_version,
                                                         )
        status = ModelVersionStatus.from_string(model_version_details.status)
        print("Model status: %s" % ModelVersionStatus.to_string(status))
        if status == ModelVersionStatus.READY:
            client.transition_model_version_stage(
              name=model_name,
              version=model_version,
              stage=stage,
            )
            break
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifactPath', type=str,
                        help='Name of mlflow artifact path location to drop model.')
    parser.add_argument('trackingURI', type=str,
                        help='mlflow host and port.')
    parser.add_argument('--outputTestData', type=str,
                        help='Name of output csv file if writing split test data.')
    args = parser.parse_args()

    artifact_path = args.artifactPath
    tracking_uri = args.trackingURI

    # Load a standard machine learning dataset
    cancer = load_breast_cancer()

    df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
    df['target'] = cancer['target']

    # Optionally write out a subset of the data, used in this tutorial for inference with the API
    if args.outputTestData:
        train, test = train_test_split(df, test_size=0.2)
        del test['target']
        test.to_csv('test.csv', index=False)

        features = [x for x in list(train.columns) if x != 'target']
        x_raw = train[features]
        y_raw = train['target']
    else:
        features = [x for x in list(df.columns) if x != 'target']
        x_raw = df[features]
        y_raw = df['target']

    # Split data into training and testing
    x_train, x_test, y_train, y_test = train_test_split(x_raw, y_raw,
                                                        test_size=.20,
                                                        random_state=123,
                                                        stratify=y_raw)

    # Build a classifier sklearn pipeline
    clf = RandomForestClassifier(n_estimators=100,
                                 min_samples_leaf=2,
                                 class_weight='balanced',
                                 random_state=123)

    preprocessor = Pipeline(steps=[('scaler', StandardScaler())])

    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('randomforestclassifier', clf)])

    # Train the model
    model.fit(x_train, y_train)

    # Grab some metrics
    accuracy_train = model.score(x_train, y_train)
    accuracy_test = model.score(x_test, y_test)

    def overwrite_predict(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return [round(x, 4) for x in result[:, 1]]
        return wrapper

    # Overwriting the model to use predict to output probabilities
    model.predict = overwrite_predict(model.predict_proba)

    # Set up mlflow tracking params for the registry
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("my-experiment")

    client = MlflowClient()

    # Start a run in the experiment and save and register the model and metrics
    with mlflow.start_run() as run:
        run_num = run.info.run_id
        model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_num, artifact_path=artifact_path)

        mlflow.log_metric('accuracy_train', accuracy_train)
        mlflow.log_metric('accuracy_test', accuracy_test)

        mlflow.sklearn.log_model(model, artifact_path)

        mlflow.register_model(model_uri=model_uri,
                              name=artifact_path)

    # Grab this latest model version
    model_version_infos = client.search_model_versions("name = '%s'" % artifact_path)
    new_model_version = max([model_version_info.version for model_version_info in model_version_infos])

    # Add a description
    client.update_model_version(
      name=artifact_path,
      version=new_model_version,
      description="Random forest scikit-learn model with 100 decision trees."
    )

    # Necessary to wait to version models
    try:
        # Move the previous model to None version
        wait_model_transition(artifact_path, int(new_model_version)-1, "None")
    except:
        pass

    # Move the latest model to Staging (could also be Production)
    wait_model_transition(artifact_path, new_model_version, "Staging")


if __name__ == "__main__":
    main()
