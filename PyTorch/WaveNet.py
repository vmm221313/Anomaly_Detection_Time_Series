import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model_zoo.WaveNet import WaveNet
from config import WaveNet_args
from utils.trainer import Trainer

from utils.dataloader import dataset


class Args():
    batch_size = 16
    num_epochs = 100
    weight_decay = 0.0001
    learning_rate = 0.01
    device = torch.device('cuda:0' if torch.cuda. is_available() else 'cpu')
    val_seq_len = 7
    train_seq_len = 30 # if you change this also change the num_dcconv_layers (see formula)
    conv_kernel_size = 4
    conv_out_channels = 32
    dilation_rates = [2**i for i in range(4)]*2 #receptive_field = kernel_size*2**(num_dcconv_layers-1)
    conv_1x1_channels = 16


args = Args()

ds = dataset('data/energy_daily_train.csv', args)

train_dataloader = DataLoader(ds, batch_size=args.batch_size)

for data, target in train_dataloader:
    print(data.shape)
    print(target.shape)
    break

from model_zoo.custom_layers.causal_conv import CausalConv1d

layers = []
layers.append(CausalConv1d(in_channels=1, out_channels=args.conv_out_channels, kernel_size=args.conv_kernel_size, dilation=args.dilation_rates[0]).to(args.device)) #because the input only has 1 channel
for dilation in args.dilation_rates[1:]:
    layers.append(CausalConv1d(in_channels=args.conv_out_channels, out_channels=args.conv_out_channels, kernel_size=args.conv_kernel_size, dilation=dilation).to(args.device))

stackedConv = nn.Sequential(*layers)
lin1 = nn.Linear(args.conv_out_channels, 1).to(args.device)
tanh = nn.Tanh().to(args.device)
lin2 = nn.Linear(args.train_seq_len, args.val_seq_len).to(args.device)
dropout = nn.Dropout()

out = stackedConv(data.view(args.batch_size, 1, args.train_seq_len)).transpose(1, 2)
out.shape

out = lin1(out)
out.shape

out = tanh(out).squeeze()
out = lin2(out)
out.shape

# +
#p = (f-1)/2 for same padding
# -

data.shape



conv_1x1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=1)
out = conv_1x1(data.view(args.batch_size, 1, args.train_seq_len))
out = relu(out)
out.shape

# +
skips = []
layers = []
relu = nn.ReLU()
l = 1
for d_r in dilation_rates:
    
    #1x1 Conv layer - equivalent to time distributed dense
    if l = 1:
        layers.append(nn.Conv1d(in_channels=1, out_channels=args.conv_1x1_channels, kernel_size=1))
    else: 
        layers.append(nn.Conv1d(in_channels=args.conv_out_channels, out_channels=args.conv_1x1_channels, kernel_size=1))

    
# -







args.dilation_rates

# +

    

# -















torch.manual_seed(31415)

args = vanillaWaveNet_args()

model = vanillaWaveNet(args).to(args.device)

model

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

trainer = Trainer(model, criterion, optimizer, args)

trainer.train(print_losses=True, plot_losses=True, plot_predictions=True)




