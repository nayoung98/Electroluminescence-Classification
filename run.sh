# 모듈 모델 학습 (supervised learning)
python train.py --ckpt_dir 'checkpoint 저장 경로' --num_epoch 50 --device 0 --model_name '모델명'

# 셀 모델 학습 (semi-supervised learning)
python train_cell.py --ckpt_dir 'checkpoint 저장 경로' --num_epoch 50 --device 0 --model_name '모델명'

# 모듈 모델 추론 
python test.py --ckpt_dir 'checkpoint 저장 경로' --data_mode 'first' --phase 'test' --device 0 --load_path 'checkpoint 저장 경로' --load_best_epoch '저장된 epoch' --batch_size 4 --model_name '모델명'

# 셀 모델 추론 
python test_cell.py --ckpt_dir 'checkpoint 저장 경로' --data_mode 'first' --phase 'test' --device 0 --load_path 'checkpoint 저장 경로' --load_best_epoch '저장된 epoch' --model_name '모델명'

# 모듈 모델 Grad-CAM
python cam_module.py --ckpt_dir 'checkpoint 저장 경로' --data_mode 'first' --phase 'test' --device 0 --load_path 'checkpoint 저장 경로' --load_best_epoch '저장된 epoch' --model_name '모델명'

# 셀 모델 Grad-CAM
python cam_cell.py --ckpt_dir 'checkpoint 저장 경로' --data_mode 'first' --phase 'test' --device 0 --load_path 'checkpoint 저장 경로' --load_best_epoch '저장된 epoch' --model_name '모델명'
