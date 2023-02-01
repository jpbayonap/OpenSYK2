#!/bin/bash
#SBATCH --job-name=parallel_job_OPENSYK # Job name
#SBATCH --nodes=1                    # Run all processes on a single node	
#SBATCH --ntasks=8                # Number of processes
#SBATCH --time=78:00:00              # Time limit hrs:min:sec
#SBATCH --partition=pepyc        #partition requested 
#SBATCH --nodelist=epyc1
#SBATCH --output=1219_%j.log # Standard output and error log
#SBATCH --array=0-29


# module load python/3
python /home/j-bayona/Git/python/OpenSYK_diag.py $SLURM_ARRAY_TASK_ID 
