#!/bin/bash -l

#SBATCH --partition=gpu-standard
#SBATCH --job-name=gan
#SBATCH --output=%x.out
#SBATCH --error=%x.err
#SBATCH --nodes=1
#SBATCH --mem=4G
#SBATCH --ntasks=1
#SBATCH --qos=normal
#SBATCH --nodelist=gpu02

#SBATCH --mail-type=ALL
#SBATCH --mail-user=s2003824@siswa.um.edu.my
#SBATCH --gres=gpu:gtx1080ti:1
~
# Loading Required module
module purge
module load miniconda/miniconda3
module load cudnn/cudnn-8.1.1/cuda-11.2
#source activate ~/conda_env
#source venv/bin/activate #library not installed yet
python proj_files/first_test/basic_main.py
