#!/bin/bash

mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifact-root --host 0.0.0.0
