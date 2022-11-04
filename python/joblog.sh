#!/bin/bash
#SBATCH --job-name=parallel_job_OPENSYK # Job name
#SBATCH --nodes=1                    # Run all processes on a single node	
#SBATCH --ntasks=24                  # Number of processes
#SBATCH --time=78:00:00              # Time limit hrs:min:sec
#SBATCH --partition=pphuket         #partition requested 
#SBATCH --nodelist=phuket3
#SBATCH --output=multiprocess_%j.log # Standard output and error log



# module load python/3
python /home/j-bayona/Git/python/OpenSYK_mainOP.py
