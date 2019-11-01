#!/bin/bash
#
##SBATCH --nodes=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=5:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=twitterRunTest
#SBATCH --mail-type=END
##SBATCH --mail-user=wj419@nyu.edu
#SBATCH --output=slurm_%j.out

module load python3/intel/3.6.3
module load pytorch/python3.6/0.3.0_4

python3 ./main.py --data ./data --epochs 1