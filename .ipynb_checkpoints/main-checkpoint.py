import torch
import torch.nn as nn
import torch.optim as optim

from model_zoo.cnn import baseCNN
from utils.trainer import Trainer
from config import CNN_args

torch.manual_seed(31415)

args = CNN_args()

model = baseCNN(args).to(args.device)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

trainer = Trainer(model, criterion, optimizer, args)

trainer.train(print_losses=True)


















from ax import optimize


def trainer_funk(params):
    class Args():
        train_seq_len = params['train_seq_len']
        val_seq_len = params['val_seq_len']
        conv_out_channels = params['conv_out_channels']
        conv_kernel_size = params['conv_kernel_size']
        max_pool_kernel_size = params['max_pool_kernel_size']
        lr = params['lr']
        wd = params['wd']
        
    args = Args()    
    args.num_epochs = 100
    args.batch_size = 16
    args.device = torch.device('cuda:0' if torch.cuda. is_available() else 'cpu') 
    args.train_dl = DataLoader(dataset('data/energy_daily_train.csv', args), batch_size=args.batch_size)
    args.val_dl = DataLoader(dataset('data/energy_daily_val.csv', args), batch_size=args.batch_size)
    args.test_dl = DataLoader(dataset('data/energy_daily_test.csv', args), batch_size=args.batch_size)

    model = baseCNN(args).to(args.device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
    trainer = Trainer(model, criterion, optimizer, args)
    
    return trainer.train()


params = {'train_seq_len':30, 
          'val_seq_len': 7,
          'conv_out_channels': 8,
          'conv_kernel_size': 4,
          'max_pool_kernel_size': 2,
          'lr': 1e-2,
          'wd': 0.01
         }

trainer_funk(params)

best_parameters, best_values, _, _ = optimize(
    parameters=[{'name': 'train_seq_len', 'type': 'choice', 'values': [14, 21, 28, 42, 63], 'value_type': 'int'},
                {'name': 'val_seq_len', 'type': 'choice', 'values': [7, 14, 21], 'value_type': 'int'},
                {'name': 'conv_out_channels', 'type': 'choice', 'values': [8, 16, 32, 64], 'value_type': 'int'},
                {'name': 'conv_kernel_size', 'type': 'choice', 'values': [3, 5, 7], 'value_type': 'int'},
                {'name': 'max_pool_kernel_size', 'type': 'choice', 'values': [3, 5, 7], 'value_type': 'int'},
                {'name': 'lr', 'type': 'range', 'bounds': [0.0001, 0.1], 'value_type': 'float'},
                {'name': 'wd', 'type': 'range', 'bounds': [0.01, 10], 'value_type': 'float'}],
    evaluation_function=trainer_funk,
    total_trials=50,
    minimize=True)

print(best_parameters)

best_values


