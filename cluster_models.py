import os
import torch

from utils.dataloader import dataset

base_path = 'data/seasons_separately/'

summer_cluster = os.listdir(base_path + 'summer/')

files = []
for f in summer_cluster:
    if f.endswith('.csv'):
        files.append(base_path + 'summer/' + f)


class Args():
    batch_size = 16
    num_epochs = 100
    weight_decay = 0.01
    learning_rate = 1e-2
    device = torch.device('cuda:0' if torch.cuda. is_available() else 'cpu')
    val_seq_len = 7
    train_seq_len = 30
    conv_kernel_size = 4
    conv_out_channels = 8
    max_pool_kernel_size = 2


args = Args()

ds = dataset(files, args)

for data, target in ds:
    print(data.shape)
    print(target.shape)
    break




