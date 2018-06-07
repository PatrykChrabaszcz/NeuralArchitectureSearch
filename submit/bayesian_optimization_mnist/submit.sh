#!/bin/bash

#SBATCH -p meta_gpu-black
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 2
#SBATCH -D /home/chrabasp/Workspace/NeuralArchitectureSearch
#SBATCH -o /home/chrabasp/Workspace/NeuralArchitectureSearch/meta_logs_mnist/o_log_%A.txt
#SBATCH -e /home/chrabasp/Workspace/NeuralArchitectureSearch/meta_logs_mnist/e_log_%A.txt

source /home/chrabasp/Workspace/env/bin/activate
export LD_LIBRARY_PATH=/home/chrabasp/cuda-8.0/lib64/

echo 'HostName' $HOSTNAME
echo 'Device' $CUDA_VISIBLE_DEVICES

python bayesian_optimization.py     --ini_file config/mnist_dataset/mnist_rnn.ini \
                                    --budget 5 \
                                    --working_dir /home/chrabasp/EEG_Results/MNIST_TEST_BO \
                                    --is_master ${IS_MASTER} \
                                    --random_fraction 0.2
