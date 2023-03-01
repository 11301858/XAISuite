#!/bin/sh

source import.sh

if [[ "$1" == "--model" && "$2" != "--import" ]]
then 
    model=$2
elif [[ "$2" == "--import" ]]
then
    echo Importing models is not supported yet
    exit 1
else
    echo No model provided. Be sure to use the --model tag immediately after --train to specify a model
    exit 1
fi

if [[ "$3" == "--data" ]]
    then 
        data=$4
else
    echo No data provided. Be sure to use the --data tag immediately after --model or --import to specify a dataset
    exit 1
fi

if [[ "$5" == "--target" ]]
    then 
        target=$6
else
    echo No target variable provided provided. Be sure to use the --target tag immediately after --data or --import to specify a target variable
    exit 1
fi

python -u CLIRunner.py "$model" "$data" "$target"
