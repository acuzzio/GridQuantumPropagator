#!/bin/bash

if [[ -z $1 ]];then
   echo ""
   ls
   echo -e "\nWrite:\n\n$0 foldername"
   exit
fi

fol=$1

#ffmpeg -r 30 -f image2 -s 1920x1080 -i ${fol}/Gaussian%04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p $fol.mp4 
ffmpeg -r 30 -f image2 -i ${fol}/Gaussian%04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p $fol.mp4 

vlc $fol.mp4
