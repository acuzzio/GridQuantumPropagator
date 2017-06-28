#!/bin/bash

source activate quantumpropagator
source ~/.gaussian

for j in 6 5 4 3 2 1
do
   for i in $(seq 1 $j)
   do
       echo -e "doing TDMZZ_${j}_${i}"
       OxiallylDensityWriter.py -g oxyallylTStemplate.fchk -t /home/alessio/Desktop/PERICYCLIC/i-Oxiallyl/01-SinglePoint/TDMZZ_${j}_${i} -o OUTPUT${j}_${i}.fchk
       cubegen 0 fdensity=SCF OUTPUT${j}_${i}.fchk Density_${j}_${i}.cube -4 h
       sed -i '3s/1$//' Density_${j}_${i}.cube
   done
done
