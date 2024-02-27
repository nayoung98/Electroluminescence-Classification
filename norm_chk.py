import pandas as pd
import numpy as np

import torchvision.transforms as transforms

from dataset import *
from torch.utils.data import DataLoader
from tqdm import tqdm

def img_norm_df(img_tensor):
    # print(img_tensor.shape)
    # img_tensor = img_tensor.numpy() # (16, 3, 977, 459)
    # print(img_tensor.shape) 
    # mean = np.mean(img_tensor, axis=(0,2,3))
    # std = np.std(img_tensor, axis=(0,2,3))
    
    img_tensor = img_tensor.permute(1,0,2,3).numpy() # (c, b, w, h)
    # print(img_tensor.shape)
    mean = np.mean(img_tensor, axis=(1,2,3))
    std = np.std(img_tensor, axis=(1,2,3))
    # print(mean.shape)
    return mean, std

dataset = ELmoduleCustomDataset(data_dir='/home/pink/nayoung/el/datasets/el', data_mode='first', phase='train', size_label=4)
loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=8)

# img_tensor_set = train_dataset['img_tensor']

mean_0, mean_1, mean_2 = [], [], []
std_0, std_1, std_2 = [], [], []

for i, data in enumerate(tqdm(loader, 0)):
# for img_tensor in tqdm(range(len(img_tensor_set))):

    mean, std = img_norm_df(data['img_tensor'])
    
    mean_0.append(mean[0])
    mean_1.append(mean[1])
    mean_2.append(mean[2])
    
    std_0.append(std[0])
    std_1.append(std[1])
    std_2.append(std[2])

print(f'이미지 평균 : {np.mean(mean_0)}, {np.mean(mean_1)}, {np.mean(mean_2)}')
print(f'이미지 표준편차 : {np.mean(std_0)}, {np.mean(std_1)}, {np.mean(std_2)}')
# print(len(mean_list))
# print(len(std_list))

# print(std_list[0])
# std_mean = np.mean(np.mean(std_list))
# std_df = pd.DataFrame(std_list)
# std_df.to_csv('./std_df.csv', index=False)
# print('Saving the std dataframe.')
# print(f'img_mean : ({np.mean(mean_list)}, img_std : {np.mean(std_list)})')