#!/bin/bash
for i in $(seq 1 15); do
echo $i
./knn_cross_validation.py mpeg7_1400_ems.pkl $i
done
