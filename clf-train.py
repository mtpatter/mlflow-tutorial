#!/usr/bin/env python

"""Example for training a random forest classifier in sklearn
   and using mlflow to save a model.
"""

import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('modelPath', type=str,
                        help='Name of mlflow artifact path location to drop model.')
    parser.add_argument('--outputTestData', type=str,
                        help='Name of output csv file if writing split test data.')
    args = parser.parse_args()

    model_path = args.modelPath

    # Load a standard machine learning dataset
    cancer = load_breast_cancer()

    df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
    df['target'] = cancer['target']

    # Define features and target variables
    features = [x for x in list(df.columns) if x != 'target']
    x_raw = df[features]
    y_raw = df['target']

    # Split data into training and testing
    x_train, x_test, y_train, _ = train_test_split(x_raw, y_raw,
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

    def overwrite_predict(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return [round(x, 4) for x in result[:, 1]]
        return wrapper

    # Overwriting the model to use predict to output probabilities
    model.predict = overwrite_predict(model.predict_proba)

    # Save the model locally
    try:
        mlflow.sklearn.save_model(model, model_path)
        print(f"Model saved at path: {model_path}")

    except Exception as e:
        print(f"Error saving model at path {model_path}: {e}. Does it already exist?")


if __name__ == "__main__":
    main()
