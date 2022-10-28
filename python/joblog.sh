#!/bin/bash
#SBATCH --job-name=parallel_job_OPENSYK # Job name
#SBATCH --nodes=1                    # Run all processes on a single node	
#SBATCH --ntasks=10                   # Number of processes
#SBATCH --time=58:00:00              # Time limit hrs:min:sec
#SBATCH --partition=pcebu           #partition requested 
#SBATCH --output=multiprocess_%j.log # Standard output and error log


# module load python/3

python /home/j-bayona/Git/python/OpenSYK_mainOP.py
