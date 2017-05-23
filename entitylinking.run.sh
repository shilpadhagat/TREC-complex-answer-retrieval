#!/bin/bash


echo "===================================================================="
echo "Generating cache for entity linking ..............                  "
echo "===================================================================="

python tc_generate_document_cache.py $1 $2 $3 enhanced


echo "===================================================================="
echo "Generating results file with entity linking and RM expansion.....   "
echo "===================================================================="

python tc_generate_entitylink_rm_cache_results.py $4 $5 $6 $7 $8 $9


echo "===================================================================="
echo "Running evaluation framework on results with entity linking.....    "
echo "===================================================================="

python eval_framework.py $10 $11
