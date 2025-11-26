#!/bin/bash
#------- qsub option -----------
#PBS -P TESTPJ01
#PBS -q B_S
#PBS -l select=1
#PBS -l walltime=12:00:00

#------- Program execution -----------
module load openmpi/5.0.7/rocm6.3.3
cd ${python3 heloo.py}
./a.out