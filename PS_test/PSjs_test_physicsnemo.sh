#!/bin/bash

#------- qsub option -----------

###ProjectID###
#PBS -P NIFS25KISC015

#PBS -q B_dev
#PBS -l select=1
#PBS -l walltime=10:00
#PBS -N TestPhysicsNeMo
#PBS -j oe

#------- Program execution -----------
cd ${PBS_O_WORKDIR}

. /data/t-shimizu/workspace/python3/neuralop_test_semi/.venv_3_12/bin/activate
python /data/t-shimizu/workspace/python3/neuralop_test_semi/physicsnemo/examples/cfd/darcy_fno/train_fno_darcy.py
deactivate
