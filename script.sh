#Train
python main.py \
  --config config/model-lm-vae.json \
  --data_root /media/cxnam/NewVolume/MEAD \
  --data_file ./assets/datas_norm/M003.txt \
  --suffix M003_lm_vae_v7_2 \
  --model model_lm_vae.model_v7_2.Model \
  --dataset model_lm_vae.dataset_v8.Dataset \
  --log_samples ./assets/samples/M003/samples_lm_vae_v7_2 \
  --skip-train-val \
  --epochs 200 \
  --pretrained /home/cxnam/Documents/MyWorkingSpace/LM2F_VAE/assets/checkpoints/checkpoint_M003_lm_vae_v9_checkpoint_200.pt \
  --n_folders 5 \

#Train GAN
python main.py \
  --config config/model-lm-vae-gan.json \
  --data_root /media/cxnam/NewVolume/MEAD \
  --data_file ./assets/datas_norm/M003.txt \
  --suffix M003_lm_vae_v8_2 \
  --model model_lm_vae.model_v8_2.Model \
  --dataset model_lm_vae.dataset_v8.Dataset \
  --log_samples ./assets/samples/M003/samples_lm_vae_v8_2 \
  --skip-train-val \
  --epochs 200 \
  --pretrained /home/cxnam/Documents/MyWorkingSpace/LM2F_VAE/assets/checkpoints/checkpoint_M003_lm_vae_v7_checkpoint_200.pt \
  --n_folders 5 \
  

#Eval
python main.py \
  --config config/model-lm-vae-gan.json \
  --data_root /media/cxnam/NewVolume/MEAD \
  --data_file ./assets/datas_norm/M003.txt \
  --suffix M003_lm_vae_v8 \
  --model model_lm_vae.model_v8.Model \
  --dataset model_lm_vae.dataset_v8.Dataset \
  --log_samples /home/cxnam/Documents/MyWorkingSpace/LM2F_VAE/assets/samples/M003/samples_lm_vae_v8 \
  --skip-train-val \
  --pretrained /home/cxnam/Documents/MyWorkingSpace/LM2F_VAE/assets/checkpoints/checkpoint_M003_lm_vae_v8_checkpoint_300.pt \
  --evaluation
