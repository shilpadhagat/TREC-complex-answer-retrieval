#!/bin/bash

echo "============================================================="
echo "Generating results file with BM25 (Baseline approach)........"
echo "============================================================="

python tc_generate_entitylinking_results.py $1 $2 $3 notenhanced

echo "============================================================="
echo "Running evaluation framework on results                      "
echo "============================================================="

python eval_framework.py $4 $5
