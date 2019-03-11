#!/bin/sh -l

#PBS -l walltime=120:00:00
#PBS -l nodes=1:ppn=8
#PBS -l mem=4gb
#PBS -l file=8gb
#PBS -N runpconsc4
#PBS -o runpconsc4.out
#PBS -e runpconsc4.err

# Load Python and python modules
module purge
module load TensorFlow/1.5.0-intel-2017b-Python-3.6.3

# Move to project folder
cd $HOME/HGF/data

# Run
python run_pconsc4.py