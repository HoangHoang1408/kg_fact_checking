#!/bin/bash
export PYTHONPATH="$(pwd)":$PYTHONPATH
poetry run python workflow/data_process/process_kg_to_trie.py \
    --tokenizer-path /Users/phamhoang1408/Desktop/graph_checking/tokenizer \
    --data-folder-path /Users/phamhoang1408/Desktop/graph_checking/data \
    --output-dir /Users/phamhoang1408/Desktop/graph_checking/trie \
    --end-sequence "</entity>"
    
