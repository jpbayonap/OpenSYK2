#!/bin/bash
#SBATCH --job-name=Test # Job name
#SBATCH --nodes=1                    # Run all processes on a single node	
#SBATCH --ntasks=1                # Number of processes
#SBATCH --time=UNLIMITED              # Time limit hrs:min:sec
#SBATCH --partition=pepyc         #partition requested 
#SBATCH --nodelist=epyc1
#SBATCH --output=1214_TESTING_%j.log # Standard output and error log



# module load python/3
python /home/j-bayona/Git/python/OpenSYK_diag.py -1