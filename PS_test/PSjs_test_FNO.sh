#!/bin/bash

#------- qsub option -----------

###ProjectID###
#PBS -P NIFS25KISC015

#PBS -q B_dev
#PBS -l select=1
#PBS -l walltime=10:00
#PBS -N TestFNO
#PBS -j oe

#------- Program execution -----------
cd ${PBS_O_WORKDIR}

. /data/t-shimizu/workspace/python3/neuralop_test_semi/.venv_neuralop/bin/activate
python -m torch.utils.collect_env
python /data/t-shimizu/workspace/python3/neuralop_test_semi/plot_FNO_darcy.py
deactivate
