#!/bin/bash

if [[ -z $2 ]];then
   echo "$0 takes two folder names as arguments: LiH000 LiH001"
   exit
fi

fn1=$1
fn2=$2

proj=${fn1}vs${fn2}
fold=confronti/


out=${fold}/${proj}/${proj}.out

a=$(grep -A20 "OVERLAP MATRIX FOR" $out | tail -10 | awk '{print $1,$2,$3,$4,$5,$6,$7,$8,$9}' | awk '{ for (i=1; i<=NF; i++) if (NR >= 1 && NR == i) print $(i - 0) }' | awk '{if ($1 < 0) {print -1} else {print 1}}' | tr '\n' ' ')
echo $a



