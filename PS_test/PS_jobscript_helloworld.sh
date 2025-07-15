#!/bin/bash

#------- qsub option -----------

###ProjectID###
#PBS -P NIFS25KISC015

#PBS -q B_dev
#PBS -l select=1
#PBS -l walltime=1:00
#PBS -N HelloWorld
#PBS -j oe

#------- Program execution -----------
cd ${PBS_O_WORKDIR}
python --version
which python
python ./helloworld.py

. .venv_3_9/bin/activate
python --version
which python
python ./helloworld.py
deactivate

. .venv_3_12/bin/activate
python --version
which python
python ./helloworld.py
deactivate
