import torch
import torch.nn as nn
import torch.optim as optim


class baseCNN(nn.Module):
    def __init__(self, args):
        super(baseCNN, self).__init__()
        
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=args.conv_out_channels, kernel_size=args.conv_kernel_size).to(args.device)
        self.max_pool = nn.MaxPool1d(kernel_size=args.max_pool_kernel_size, stride=1).to(args.device)
        self.lin = nn.Linear(args.conv_out_channels*((args.train_seq_len-1*(args.conv_kernel_size-1))-1*(args.max_pool_kernel_size-1)), args.val_seq_len).to(args.device)
        
    def forward(self, x):
        out = self.conv1d(x)
        out = self.max_pool(out)
        out = torch.flatten(out, start_dim=1, end_dim=2)
        #print(out.shape)
        #print(self.lin)
        out = self.lin(out)
        
        return out
