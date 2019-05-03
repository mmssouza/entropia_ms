#!/bin/bash
for i in $(seq 1 16); do
echo $i
./knn_cross_validation.py kimia99_ems.pkl $i
done
