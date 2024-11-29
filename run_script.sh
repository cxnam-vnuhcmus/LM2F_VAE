#!/bin/bash

### Comment lines start with ## or #+space
### Slurm option lines start with #SBATCH

### Here are the SBATCH parameters that you should always consider:
##SBATCH --time=0-00:05:00   ## days-hours:minutes:seconds
#SBATCH --mem 6000M         ## 3GB ram (hardware ratio is < 4GB/core)
#SBATCH --ntasks=1          ## Not strictly necessary because default is 1
#SBATCH --cpus-per-task=10   ## Use greater than 1 for parallelized jobs
#SBATCH --gpus=1

### Here are other SBATCH parameters that you may benefit from using, currently commented out:
#SBATCH --job-name=LM2F ## job name
#SBATCH --output=job.out  ## standard out file

# source activate myenv

python main.py \
  --config config/model-lm-vae.json \
  --data_root /media/cxnam/MEAD \
  --data_file /media/cxnam/LM2F_VAE/assets/datas_norm/M003.txt \
  --suffix M003_lm_vae_v43 \
  --model model_lm_vae.model_v4_3.Model \
  --dataset model_lm_vae.dataset_v7.Dataset \
  --log_samples /media/cxnam/LM2F_VAE/assets/samples/M003/samples_lm_vae_v43 \
  --skip-train-val \
  --epochs 300

echo 'finished'
