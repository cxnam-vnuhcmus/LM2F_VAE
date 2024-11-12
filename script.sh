#Train
python main.py \
  --config config/model-lm-vae.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas_norm/M003.txt \
  --suffix M003_lm_vae_v6 \
  --model model_lm_vae.model_v6.Model \
  --dataset model_lm_vae.dataset_v7.Dataset \
  --log_samples /home/cxnam/Documents/MyWorkingSpace/LM2F_VAE/assets/samples/M003/samples_lm_vae_v6 \
  --skip-train-val \
  --epochs 300 \
  --pretrained /home/cxnam/Documents/MyWorkingSpace/LM2F_VAE/assets/checkpoints/checkpoint_M003_lm_vae_v4_checkpoint_50.pt \
  --n_folders 5 \
  

#Eval
python main.py \
  --config config/model-lm-vae.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas_norm/M003.txt \
  --suffix M003_lm_vae_v5 \
  --model model_lm_vae.model_v5.Model \
  --dataset model_lm_vae.dataset_v5.Dataset \
  --log_samples /home/cxnam/Documents/MyWorkingSpace/LM2F_VAE/assets/samples/M003/samples_lm_vae_v5 \
  --skip-train-val \
  --pretrained /home/cxnam/Documents/MyWorkingSpace/LM2F_VAE/assets/checkpoints/best_M003_lm_vae_v5_checkpoint_1_MSE=-0.0024.pt \
  --evaluation \
  --n_folders 10
