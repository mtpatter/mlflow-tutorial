#!/bin/bash

mlflow models serve -m $1 -p 1234 -h 0.0.0.0 --env-manager local
