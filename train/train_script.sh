# [cell_max] el, devices_id = 0, 1, 2
python train_cell.py --ckpt_dir '/home/pink/nayoung/el/main/checkpoints/240103_resnet34att_max' --num_epoch 100 --device 0 --multi_gpu 'on'

# [cell_avg] el2, devices_id = 3, 4, 5
python train_cell.py --ckpt_dir '/home/pink/nayoung/el/main/checkpoints/240103_resnet34att_avg' --num_epoch 100 --device 3 --multi_gpu 'on'

# [loss] el, devices_id = 0, 1, 2
python train_cell.py --ckpt_dir '/home/pink/nayoung/el/main/checkpoints/240104_resnet34_loss' --num_epoch 100 --device 0 --multi_gpu 'on'
# [loss_mean] el, devices_id = 0, 1, 2
python train_cell.py --ckpt_dir '/home/pink/nayoung/el/main/checkpoints/240105_resnet34_loss_mean' --num_epoch 100 --device 0 --multi_gpu 'on'
# [loss_non_fault] el2, devices_id = 3, 4, 5
python train_cell.py --ckpt_dir '/home/pink/nayoung/el/main/checkpoints/240105_resnet34_loss_non' --num_epoch 100 --device 3 --multi_gpu 'on'
# [loss_non_fault_cnt] el, devices_id = 0, 1, 2
python train_cell.py --ckpt_dir '/home/pink/nayoung/el/main/checkpoints/240111_resnet34_loss_non_cnt' --num_epoch 100 --device 0 --multi_gpu 'on'

# [cell_base] el, devices_id = 0, 1, 2
python train_cell.py --ckpt_dir '/home/pink/nayoung/el/main/checkpoints/240109_resnet34_cell' --num_epoch 100 --device 0 --multi_gpu 'on'

# [Base model_crop, ResNet34] el2
python train.py --ckpt_dir '/home/pink/nayoung/el/main/checkpoints/240109_resnet34_crop' --num_epoch 100 --device 3 --batch_size 4
# [Base model_non crop] el3
python train.py --ckpt_dir '/home/pink/nayoung/el/main/checkpoints/240109_resnet34_noncrop' --num_epoch 100 --device 3 --batch_size 4
# [Base model_crop, ResNext50] el2
python train.py --ckpt_dir '/home/pink/nayoung/el/main/checkpoints/240111_resnext50_crop' --num_epoch 100 --device 4 --batch_size 4

# [Contrastive] el3, 3
python train_cell.py --ckpt_dir '/home/pink/nayoung/el/main/checkpoints/240112_resnet34_cont' --num_epoch 100 --device 3 --multi_gpu 'on' --batch_size 2

# [loss_non_fault_cnt + attention module] el, devices_id = 0, 1, 2
python train_cell.py --ckpt_dir '/home/pink/nayoung/el/main/checkpoints/240114_resnet34_loss_non_cnt_att' --num_epoch 100 --device 0 --multi_gpu 'on'
# [loss_non_fault + attention module] el2, devices_id = 3, 4, 5
python train_cell.py --ckpt_dir '/home/pink/nayoung/el/main/checkpoints/240114_resnet34_loss_non_att' --num_epoch 100 --device 3 --multi_gpu 'on'

# [loss_non_fault_cnt + attention module] el, devices_id = 0, 1, 2
python train_cell.py --ckpt_dir '/home/pink/nayoung/el/main/checkpoints/240116_resnet34_loss_non_cnt_att' --num_epoch 100 --device 0 --multi_gpu 'on'

# [Tensorboard]
tensorboard --logdir='/home/pink/nayoung/el/main/checkpoints/240111_resnet34_loss_non_cnt' --port=6007