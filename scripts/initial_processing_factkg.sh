#!/bin/bash

export PYTHONPATH="$(pwd)":$PYTHONPATH
poetry run python workflow/data_process/initial_processing_factkg.py \
    --data-folder-path /Users/phamhoang1408/Desktop/graph_checking/data \
