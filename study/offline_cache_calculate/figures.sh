#!/bin/bash

figures_name="PA TW UK CF"

for dataset in $figures_name
do
  gnuplot -c ./cache_hit.plt "$dataset" > ${dataset}_hit_rate.eps
done
