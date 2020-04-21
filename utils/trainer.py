import numpy as np
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


class Trainer(object):
    def __init__(self, model, criterion, optimizer, args):
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
    
    def train(self, print_losses=False, plot_losses=False):
        for epoch in range(self.args.num_epochs):
            self.model.train()
            
            train_losses = []
            val_losses = []
            for data, target in self.args.train_dl:
                self.optimizer.zero_grad()
                
                pred = self.model(data.view(self.args.batch_size, 1, self.args.train_seq_len))
                loss = self.criterion(pred, target)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())
                
            
            if print_losses:
                if (epoch+1)%20 == 0: 
                    train_loss = np.array(losses).mean()
                    val_loss = self.validate()
                    val_losses.append(val_loss)
                    print('Train loss after {} epochs = {}'.format(epoch+1, train_loss))
                    print('Validation loss after {} epochs = {}'.format(epoch+1, val_loss))
                    
        if plot_losses:
            plt.plot(train_losses, color = 'red')
            plt.plot(val_losses, color = 'blue')
            plt.legend('train_losses', 'val_losses')
            plt.show()
        
        return self.validate()
                
    def validate(self):
        self.model.eval()
        with torch.no_grad():
            losses = []
            for data, target in self.args.val_dl:
                pred = self.model(data.view(self.args.batch_size, 1, self.args.train_seq_len))
                loss = self.criterion(pred, target)
                losses.append(loss.item())
            
            losses = np.array(losses)
            return losses.mean()


