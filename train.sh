python train.py --save_dir 8vc \
		--data_name againvc freevc medium styletts triaan vqmivc knnvc sigvc \
        --warmup_epochs 1 --dur_range 2 2 \
		--val_data_name vc-dev \
	    --batch_size 256 --workers 40 \
	    --mels 80 --fft 512 \
		--model ConformerMFA --embd_dim 256 \
		--classifier ArcFace --angular_m 0.2 --angular_s 32 --dropout 0 \
		--gpu 0,1,2,3 --epochs 25  --start_epoch 0 --lr 0.001 &