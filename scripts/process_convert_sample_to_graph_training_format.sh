#!/bin/bash
export PYTHONPATH="$(pwd)":$PYTHONPATH
poetry run python workflow/data_process/process_convert_sample_to_graph_training_format.py
