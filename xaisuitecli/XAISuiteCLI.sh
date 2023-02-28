#!/bin/sh


source import.sh

if [[ $# -ge 2 ]]
then
    if [[ $1 == "--check" ]] 
    then
       echo Checking...
       source check.sh ${@:2}
    elif [[ $1 == "--train" ]]
    then
        echo Training...
        source train.sh ${@:2}
    fi 
fi