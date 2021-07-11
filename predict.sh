#!/bin/bash

curl http://localhost:1234/invocations  -H 'Content-Type: text/csv' --data-binary @$1
