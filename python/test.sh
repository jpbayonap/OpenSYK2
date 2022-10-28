#!/bin/bash
#SBATCH --job-name=parallel_job_OPENSYK # Job name
#SBATCH --nodes=1                    # Run all processes on a single node	
#SBATCH --ntasks=12                   # Number of processes
#SBATCH --mem=1gb                    # Total memory limit
#SBATCH --time=00:01:00              # Time limit hrs:min:sec
#SBATCH --output=multiprocess_%j.log # Standard output and error log
date;hostname;pwd

# module load python/3

echo "Tonces que"
