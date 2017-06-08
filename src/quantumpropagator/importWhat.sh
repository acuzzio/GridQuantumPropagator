#!/bin/bash

for i in [a-zA-Z0-9]*.py
do
   a=$(grep "^def " $i | awk '{print $2}' | awk -F '(' '{print $1}' | tr '\n' ',')
   echo "from .${i%.*} import ($a)" | sed 's/,)/)/' | sed 's/,/, /g'
done

