#!/bin/bash

if [[ -z $1 ]];then
   echo ""
   ls
   echo -e "\nWrite:\n\n$0 foldername"
   exit
fi

fol=$1

ffmpeg -r 30 -f image2 -i ${fol}/Gaussian%04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p $fol.mp4 
vlc ${fol}.mp4


# Gaussian0002_state_1_0.png

#for i in 0 1 2 3 4 5 6 7
#do
#for j in 0 1 2 3 4 5 6 7
#do
#
#ffmpeg -r 30 -f image2 -i ${fol}/Gaussian%04d_state_${i}_${j}.png -vcodec libx264 -crf 25  -pix_fmt yuv420p ${fol}_state_${i}_${j}.mp4 
#
#done
#done
#vlc ${fol}_state_0_0.mp4
