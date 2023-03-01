#!/bin/bash -l

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=magical_rl_pick_ppo

# Remove one # to uncommment
#SBATCH --output=output/%x-%j.out

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=0-24:00:00   
#SBATCH --gres=gpu:1

# Turn on mail notification. There are many possible self-explaining values:
# NONE, BEGIN, END, FAIL, ALL (including all aforementioned)
# For more values, check "man sbatch"
#SBATCH --mail-type=NONE
# Remember to set your email address here instead of nobody
#SBATCH --mail-user=khanh.nguyen@princeton.edu

source ~/.bashrc
conda activate magical

home_dir=/n/fs/nlp-kn5378/code/magical-1
cd $home_dir

# Define and create a unique scratch directory for this job
#tag=pick_ppo;
#OUT_DIRECTORY='experiments/${tag}'
#mkdir ${OUT_DIRECTORY};

# Submit jobs.
exp_name=pick_ppo_${SLURM_JOB_ID}
srun --gres=gpu:1 -n 1 --mem=24G --exclusive python train_rl.py \
                                                -config configs/pick_ppo.yaml \
                                                -name $exp_name 

# Finish the script
exit 0
