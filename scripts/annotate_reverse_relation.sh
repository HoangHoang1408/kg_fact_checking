#!/bin/bash
export PYTHONPATH="$(pwd)":$PYTHONPATH
poetry run python workflow/data_annotate/annotate_revert_relation.py
