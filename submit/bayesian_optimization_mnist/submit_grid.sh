#!/bin/bash

echo $'Create log file directory...'
mkdir /home/chrabasp/Workspace/NeuralArchitectureSearch/meta_logs_mnist/
echo $'Log file directory created... \n'

echo $'Submitting master job'
sbatch --export=JOB_NUM=${0},IS_MASTER=1 submit.sh
echo $'Master job submitted \n'

for i in {1..7}
do
    echo $'Submitting worker job'
    sbatch --export=JOB_NUM=${i},IS_MASTER=0 submit.sh
    echo $'Worker job submitted\n'
done

