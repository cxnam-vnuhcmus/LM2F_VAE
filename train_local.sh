python main.py \
  --config config/model-lm-vae-gan.json \
  --data_root /media/cxnam/NewVolume/MEAD \
  --data_file ./assets/datas_norm/M003.txt \
  --suffix M003_lm_vae_v8_1 \
  --model model_lm_vae.model_v8_1.Model \
  --dataset model_lm_vae.dataset_v8.Dataset \
  --log_samples ./assets/samples/M003/samples_lm_vae_v8_1 \
  --skip-train-val \
  --epochs 200

python main.py \
  --config config/model-lm-vae-gan.json \
  --data_root /media/cxnam/NewVolume/MEAD \
  --data_file ./assets/datas_norm/M003.txt \
  --suffix M003_lm_vae_v8_2 \
  --model model_lm_vae.model_v8_2.Model \
  --dataset model_lm_vae.dataset_v8.Dataset \
  --log_samples ./assets/samples/M003/samples_lm_vae_v8_2 \
  --skip-train-val \
  --epochs 200