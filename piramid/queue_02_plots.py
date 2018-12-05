#!/bin/bash
# set the number of nodes and processes per node
#SBATCH --nodes=1

# Partition name 
#SBATCH --partition=small

# set the number of tasks (processes) per node.
#SBATCH --ntasks-per-node=1

# set max wallclock time
#SBATCH --time=45:00

# set name of job
#SBATCH --job-name=02_full

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=bruno@oac.unc.edu.ar


module load clemente
module load anaconda2/5.0.0

source activate benv3

python 02_plots_noML.py './plots_full' -s True 
