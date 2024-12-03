#!/bin/bash
export PYTHONPATH="$(pwd)":$PYTHONPATH
poetry run python workflow/data_annotate/annotate_in_out_entities.py 
