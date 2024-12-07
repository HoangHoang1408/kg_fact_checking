#!/bin/bash

export PYTHONPATH="$(pwd)":$PYTHONPATH
poetry run python workflow/data_process/initial_processing_factkg.py \
    --data-folder-path /home/namb/hoangpv4/kg_fact_checking/data \
