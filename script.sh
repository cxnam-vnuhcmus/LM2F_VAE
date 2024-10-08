#Train
python main.py \
  --config config/model-lm-vae-gan.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas_norm/M003.txt \
  --suffix M003_lm_vae_v3 \
  --model model_lm_vae.model_v3.Model \
  --dataset model_lm_vae.dataset_v1.Dataset \
  --skip-train-val \
  --log_samples /home/cxnam/Documents/MyWorkingSpace/LM2F_VAE/assets/samples/M003/samples_lm_vae_v3 \
  --epochs 100 \
  --pretrained ./assets/checkpoints/best_M003_lm_vae_v1_checkpoint_1_MSE=-0.1148.pt \
  --n_folders 5 \
  

#Eval
python main.py \
  --config config/model-lm-vae-gan.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas_norm/M003.txt \
  --suffix M003_lm_vae_v3 \
  --model model_lm_vae.model_v3.Model \
  --dataset model_lm_vae.dataset_v1.Dataset \
  --skip-train-val \
  --pretrained /home/cxnam/Documents/MyWorkingSpace/LM2F_VAE/assets/checkpoints/best_M003_lm_vae_v3_checkpoint_1_MSE=-0.0040.pt \
  --evaluation \
  --log_samples /home/cxnam/Documents/MyWorkingSpace/LM2F_VAE/assets/samples/M003/samples_lm_vae_v3 \
  --n_folders 10