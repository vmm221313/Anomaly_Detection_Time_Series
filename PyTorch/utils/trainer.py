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
    
    def train(self, print_losses=False, plot_losses=False, tuning=False, plot_predictions=False):
        train_losses_step = []
        train_losses_epoch = []
        val_losses = []
        for epoch in range(self.args.num_epochs):
            self.model.train()
            
            for data, target in self.args.train_dl:
                self.optimizer.zero_grad()
                
                pred = self.model(data.view(self.args.batch_size, 1, self.args.train_seq_len))
                loss = self.criterion(pred, target)
                loss.backward()
                self.optimizer.step()
                train_losses_step.append(loss.item())
                
            
            if (epoch+1)%20 == 0: 
                train_loss = np.array(train_losses_step).mean()
                train_losses_epoch.append(train_loss)
                val_loss = self.validate()
                val_losses.append(val_loss)

                if print_losses:
                    print('Train loss after {} epochs = {}'.format(epoch+1, train_loss))
                    print('Validation loss after {} epochs = {}'.format(epoch+1, val_loss))
                    
        if plot_losses:
            plt.plot(train_losses_epoch, color = 'red')
            plt.plot(val_losses, color = 'blue')
            plt.show()

        if tuning: # return val loss if tuning
            return self.validate()#, train_losses_epoch, val_losses
        else:
            return self.test(plot_predictions)
                
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
        
    def test(self, plot_predictions):
        self.model.eval()
        with torch.no_grad():
            losses = []
            for data, target in self.args.test_dl:
                pred = self.model(data.view(1, 1, self.args.train_seq_len))
                loss = self.criterion(pred.squeeze(), target.squeeze())
                losses.append(loss.item())
                if plot_predictions:
                    plt.plot(pred.squeeze().numpy(), color = 'red')
                    plt.plot(target.squeeze().numpy(), color = 'blue')
                    plt.show()
                    
            losses = np.array(losses)
            return losses.mean()


