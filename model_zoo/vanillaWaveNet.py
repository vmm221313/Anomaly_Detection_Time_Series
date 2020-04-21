# ls

import torch
import torch.nn as nn
from model_zoo.custom_layers.causal_conv import CausalConv1d

class vanillaWaveNet(nn.Module):
    def __init__(self, args):
        super(vanillaWaveNet, self).__init__()
        
        layers = []
        layers.append(CausalConv1d(in_channels=1, out_channels=args.conv_out_channels, kernel_size=args.conv_kernel_size, dilation=args.dilation_rates[0]).to(args.device)) #because the input only has 1 channel
        for dilation in args.dilation_rates[1:]:
            layers.append(CausalConv1d(in_channels=args.conv_out_channels, out_channels=args.conv_out_channels, kernel_size=args.conv_kernel_size, dilation=dilation).to(args.device))

        self.stackedConv = nn.Sequential(*layers)
        self.lin1 = nn.Linear(args.conv_out_channels, 1).to(args.device)
        self.tanh = nn.Tanh().to(args.device)
        self.lin2 = nn.Linear(args.train_seq_len, args.val_seq_len).to(args.device)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = self.stackedConv(x).transpose(1, 2)
        out = self.dropout(out)
        out = self.lin1(out)
        out = self.tanh(out).squeeze()
        out = self.lin2(out)

        return out
