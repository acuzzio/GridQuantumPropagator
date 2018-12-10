#!/bin/bash

if [[ -z $3 ]]; then
   echo "$0 numberstep deltaT Projectname"
   exit
else
   numberstep=$1
   deltaT=$2
   prj=$3
fi

#out=output-graphics/cincoManana
#out2=output-graphics/LongcincoManana

#python setup.py build_ext --inplace

/usr/bin/time -p python3 GridIntegrator.py -s $numberstep -d $deltaT -n $prj

#(/usr/bin/time -p python3 GridIntegrator.py -s 100 -d 0.04 -n HundredTimings)  > $out
#/usr/bin/time -p python3 GridIntegrator.py -s 100000 -d 0.04 -n TryToSeeHighMomentum > $out2

