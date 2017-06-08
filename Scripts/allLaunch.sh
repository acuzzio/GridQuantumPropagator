#!/bin/bash

label=LiH

if [[ -z $1 ]];then
   echo -e "\n\nlaunchmode:\n$0 L\n\nparsemode:\n$0 P\n\n"
   exit
fi

for i in {0..31}
do 
   ii=$(echo "$i" | awk '{printf "%03i", $1}')
   kk=$(echo "$i" | awk '{printf "%03i", $1+1}')
   n1=${label}${ii}
   n2=${label}${kk}
if [[ $1 == 'L' ]];then
./launch1.sh $n1 $n2
else
./parser.sh $n1 $n2
fi
done

