#!/bin/bash

here=$PWD

if [[ -z $2 ]];then
   echo "$0 takes two folder names as arguments: LiH000 LiH001"
   exit
fi

fn1=$1
fn2=$2

proj=${fn1}vs${fn2}
fold=confronti/


jobi1=${fn1}/${fn1}.JobIph
ox=${fn2}/${fn2}.xyz
nx=${fold}/${proj}/${proj}.xyz
ni=${fold}/${proj}/${proj}.input
njobi1=${fold}/${proj}/${proj}.JobIph.old

mkdir $fold 2> /dev/null

if [[ -d $fold/$proj ]];then
   echo "$fold/$proj exists"
   exit
else
mkdir $fold/$proj
fi

cp $ox $nx
cp $jobi1 $njobi1
#cp $jobi2 $njobi2


cat > $ni << MAFG
>> LINK FORCE \$Project.JobIph.old JOB001
>> LINK FORCE \$Project.JobIph JOB002

&Gateway
  coord=\$Project.xyz
  basis=6-31PPGDD
  group=NoSym

&Seward

&Rasscf
  nactel = 4 0 0 
  ras2 = 20
  inactive = 0 
  ciroot = 9 9 1 

&Rassi
*  mees
  NROFJOBIPHS
  2 9 9
  1 2 3 4 5 6 7 8 9
  1 2 3 4 5 6 7 8 9
  OVERLAP 

&grid_it
  all 

>> COPY \$Project.rassi.h5 \$HomeDir
>> COPY \$Project.JobIph \$HomeDir

MAFG

cd $fold/$proj
LaunchMolcas ${proj}.input
cd $here



