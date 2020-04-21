import torch
from utils.dataloader import dataset
from torch.utils.data import DataLoader

class training_args():
    def __init__(self):
        self.batch_size = 16
        self.num_epochs = 100
        self.weight_decay = 0.01
        self.learning_rate = 1e-2
        self.device = torch.device('cuda:0' if torch.cuda. is_available() else 'cpu')
        
class CNN_args(training_args):
    def __init__(self):
        super().__init__()
        self.val_seq_len = 7
        self.train_seq_len = 30
        self.conv_kernel_size = 4
        self.conv_out_channels = 8
        self.max_pool_kernel_size = 2
        self.train_dl = DataLoader(dataset('data/energy_daily_train.csv', self), batch_size=self.batch_size)
        self.val_dl = DataLoader(dataset('data/energy_daily_val.csv', self), batch_size=self.batch_size)
        self.test_dl = DataLoader(dataset('data/energy_daily_test.csv', self), batch_size=self.batch_size)
        
class vanillaWaveNet_args(training_args):
    def __init__(self):
        super().__init__()
        self.val_seq_len = 7
        self.train_seq_len = 30
        self.conv_kernel_size = 2
        self.conv_out_channels = 32
        self.dilation_rates = [2**i for i in range(4)] 
        self.train_dl = DataLoader(dataset('data/energy_daily_train.csv', self), batch_size=self.batch_size)
        self.val_dl = DataLoader(dataset('data/energy_daily_val.csv', self), batch_size=self.batch_size)
        self.test_dl = DataLoader(dataset('data/energy_daily_test.csv', self), batch_size=self.batch_size)
        


        

    
    

    