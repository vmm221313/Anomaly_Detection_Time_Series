import torch
import torch.nn as nn

class vanillaLSTM(nn.Module):
    def __init__(self, args):
        super(vanillaLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=args.hidden_size)
        self.lin = nn.Linear(args.hidden_size, args.val_seq_len)
        self.tanh = nn.Tanh()
        self.args = args
        
    def forward(self, x, testing=False):
        
        if testing:
            out, (h_n, c_n) = self.lstm(x.view(self.args.train_seq_len, 1, 1))
        else:
            out, (h_n, c_n) = self.lstm(x.view(self.args.train_seq_len, self.args.batch_size, 1))
        
        h_n = self.tanh(h_n.squeeze())
        h_n = self.lin(h_n)
        
        return h_n
