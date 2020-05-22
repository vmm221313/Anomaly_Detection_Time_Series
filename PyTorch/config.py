import torch
from utils.dataloader import dataset
from torch.utils.data import DataLoader

class training_args():
    def __init__(self):
        self.batch_size = 16
        self.num_epochs = 100
        self.weight_decay = 0.01
        self.learning_rate = 0.01
        self.device = torch.device('cuda:0' if torch.cuda. is_available() else 'cpu')

class CNN_args(training_args):
    def __init__(self):
        super().__init__()
        self.val_seq_len = 7
        self.train_seq_len = 30
        self.conv_kernel_size = 4
        self.conv_out_channels = 8
        self.max_pool_kernel_size = 2
        self.train_dl = DataLoader(dataset('data/2017_energy_5_min_train.csv', self), batch_size=self.batch_size)
        self.val_dl = DataLoader(dataset('data/2017_energy_5_min_val.csv', self), batch_size=self.batch_size)
        self.test_dl = DataLoader(dataset('data/2017_energy_5_min_test.csv', self), batch_size=1)

class vanillaWaveNet_args(training_args):
    def __init__(self):
        super().__init__()
        self.val_seq_len = 7
        self.learning_rate = 0.01
        self.train_seq_len = 30 # if you change this also change the num_dcconv_layers (see formula)
        self.conv_kernel_size = 4
        self.conv_out_channels = 32
        self.dilation_rates = [2**i for i in range(4)] #receptive_field = kernel_size*2**(num_dcconv_layers-1)
        self.train_dl = DataLoader(dataset('data/2017_energy_5_min_train.csv', self), batch_size=self.batch_size)
        self.val_dl = DataLoader(dataset('data/2017_energy_5_min_val.csv', self), batch_size=self.batch_size)
        self.test_dl = DataLoader(dataset('data/2017_energy_5_min_test.csv', self), batch_size=1)

class LSTM_args(training_args):
    def __init__(self):
        super().__init__()
        self.val_seq_len = 1
        self.train_seq_len = 30
        self.hidden_size = 200
        self.num_layers = 3
        self.learning_rate = 0.1
        self.train_dl = DataLoader(dataset('data/energy_daily_train.csv', self), batch_size=self.batch_size)
        self.val_dl = DataLoader(dataset('data/energy_daily_val.csv', self), batch_size=self.batch_size)
        self.test_dl = DataLoader(dataset('data/energy_daily_test.csv', self), batch_size=self.batch_size)

        





    

    
