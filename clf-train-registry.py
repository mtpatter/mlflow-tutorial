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


def wait_model_ready(model_name, model_version):
    client = MlflowClient()
    for _ in range(10):
        model_version_details = client.get_model_version(name=model_name,
                                                         version=model_version)
        status = ModelVersionStatus.from_string(model_version_details.status)
        print("Model status: %s" % ModelVersionStatus.to_string(status))
        if status == ModelVersionStatus.READY:
            return True
        time.sleep(1)
    return False


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

    # Define features and target variables
    features = [x for x in list(df.columns) if x != 'target']
    x_raw = df[features]
    y_raw = df['target']

    # Split data into training and testing
    x_train, x_test, y_train, y_test = train_test_split(x_raw, y_raw,
                                                    test_size=0.20,
                                                    random_state=123,  # seeding
                                                    stratify=y_raw)

    # Optionally write test data, used for inference example with the API
    if args.outputTestData:
        test_df = pd.DataFrame(data=x_test, columns=features)
        test_df.to_csv('test.csv', index=False)
        print("Test data written to 'test.csv'")

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

    # Create an input example for logging the model
    input_example = pd.DataFrame(data=x_train.iloc[:1].values, columns=x_train.columns)

    # Start a run in the experiment and save and register the model and metrics
    with mlflow.start_run() as run:
        run_num = run.info.run_id
        model_uri = f"runs:/{run_num}/{artifact_path}"

        mlflow.log_metric('accuracy_train', accuracy_train)
        mlflow.log_metric('accuracy_test', accuracy_test)

        # Log the model with input example to infer signature automatically
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            input_example=input_example  # Include the input example here
        )

        registered_model_info = mlflow.register_model(model_uri=model_uri,
                                                      name=artifact_path)

    # Get the new model version from registration response
    new_model_version = registered_model_info.version

    # Add a description to the registered model version
    client.update_model_version(
      name=artifact_path,
      version=new_model_version,
      description="Random forest scikit-learn model with 100 decision trees."
    )

    # Wait for the model to be ready before setting an alias
    if wait_model_ready(artifact_path, new_model_version):
        # Set the alias for the new model version to "Staging"
        client.set_registered_model_alias(
            name=artifact_path,
            alias="Staging",
            version=new_model_version
        )
        print(f"Set 'Staging' alias for model '{artifact_path}' \
            version {new_model_version}")

        # Verify that the alias was set correctly by checking aliases of this version.
        try:
            model_version_details = client.get_model_version(name=artifact_path,
                                                             version=new_model_version)
            print(f"Aliases for model '{artifact_path}' \
                version {new_model_version}: {model_version_details.aliases}")

            if "Staging" in model_version_details.aliases:
                print(f"Successfully verified 'Staging' alias for model \
                    '{artifact_path}' version {new_model_version}")
            else:
                print(f"Warning: 'Staging' alias not found for model \
                    '{artifact_path}' version {new_model_version}")

        except Exception as e:
            print(f"Error verifying alias: {e}")

    else:
        print("Model did not become ready in time")


if __name__ == "__main__":
    main()
