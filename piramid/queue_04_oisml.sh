#!/bin/bash
# set the number of nodes and processes per node
#SBATCH --nodes=1

# Partition name 
#SBATCH --partition=batch

#SBATCH --exclusive=user

# set the number of tasks (processes) per node.
#SBATCH --ntasks-per-node=56

# set max wallclock time
#SBATCH --time=24:00:00

# set name of job
#SBATCH --job-name=04_oisparl

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=bruno@oac.unc.edu.ar


module load clemente
module load anaconda2/5.0.0

source activate benv3

python 04_ois_ml_parallel.py ./plots/new_ml_ois_parallel -j 27 -n 4

