import torch
import torch.nn as nn
import torch.optim as optim

from model_zoo.vanillaLSTM import vanillaLSTM
from config import LSTM_args
from utils.trainer import Trainer

torch.manual_seed(31415)

args = LSTM_args()

model = vanillaLSTM(args).to(args.device)

model

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

trainer = Trainer(model, criterion, optimizer, args)

trainer.train(tuning=True, print_losses=True, plot_losses=True)







# class Args():
#     batch_size = 16
#     num_epochs = 100
#     weight_decay = 0.0001
#     learning_rate = 0.01
#     device = torch.device('cuda:0' if torch.cuda. is_available() else 'cpu')
#     val_seq_len = 1
#     train_seq_len = 14 
#     hidden_size = 100
#     num_layers = 1
#
# args = Args()
#
# ds = dataset('data/energy_daily_train.csv', args)
#
# train_dataloader = DataLoader(ds, batch_size=args.batch_size)
#
# for data, target in train_dataloader:
#     print(data.shape)
#     print(target.shape)
#     print(model(data).shape)
#     break




