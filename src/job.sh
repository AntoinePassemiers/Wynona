#!/bin/sh -l

#PBS -l walltime=120:00:00
#PBS -l nodes=1:ppn=8
#PBS -l mem=8gb
#PBS -l file=32gb
#PBS -N deepcp
#PBS -o deepcp.out
#PBS -e deepcp.err

# Load Python and python modules
module purge
module load Python/3.6.4-intel-2018a
module load matplotlib/2.1.2-intel-2018a-Python-3.6.4
module load PyTorch/0.4.1-intel-2018a-Python-3.6.4

# Move to project folder
cd $HOME/HGF/src

# Run
python main.py