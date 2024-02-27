# Data check
python test_chk.py --ckpt_dir '/home/pink/nayoung/el/main/checkpoints/230831_resnet34_4_bs8' --data_mode 'first' --phase 'test' --size_label 4 --batch_size 8

# Grad CAM
python cam.py --ckpt_dir '/home/pink/nayoung/el/main/checkpoints/240102_resnet34att_max' --data_mode 'first' --phase 'test' --device 0 --load_path '/home/pink/nayoung/el/main/checkpoints/240102_resnet34att_max/20.pth' --multi_gpu 'on'

python test.py --ckpt_dir '/home/pink/nayoung/el/main/checkpoints/240109_resnet34_noncrop' --data_mode 'first' --phase 'test' --size_label 8 --device 0 --load_path '/home/pink/nayoung/el/main/checkpoints/240109_resnet34_noncrop/best_10.pth' --batch_size 4
python test_cell.py --ckpt_dir '/home/pink/nayoung/el/main/checkpoints/240105_resnet34_loss_mean' --data_mode 'first' --phase 'test' --device 0 --load_path '/home/pink/nayoung/el/main/checkpoints/240105_resnet34_loss_mean/best_5.pth' --multi_gpu 'on'
python test_cell_chk.py --ckpt_dir '/home/pink/nayoung/el/main/checkpoints/240102_resnet34att_max' --data_mode 'first' --phase 'test' --device 0 --load_path '/home/pink/nayoung/el/main/checkpoints/240102_resnet34att_max/20.pth' --multi_gpu 'on'

python test_cell.py --ckpt_dir '/home/pink/nayoung/el/main/checkpoints/240111_resnet34_loss_non_cnt' --data_mode 'first' --phase 'test' --device 0 --load_path '/home/pink/nayoung/el/main/checkpoints/240111_resnet34_loss_non_cnt/best_7.pth' --multi_gpu 'on'
