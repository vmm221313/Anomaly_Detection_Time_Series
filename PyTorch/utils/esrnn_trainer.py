import os
import time
import numpy as np
import copy
import torch
import torch.nn as nn
from utils.esrnn_loss_modules import PinballLoss, sMAPE, np_sMAPE
#from utils.logger import Logger
import pandas as po


class ESRNNTrainer(nn.Module):
    def __init__(self, args, model, train_t, val_t, test_t):
        super(ESRNNTrainer, self).__init__()
        self.model = model.to(args.device)
        self.args = args
        #self.dl = dataloader
        self.train_t = train_t
        self.val_t = val_t
        self.test_t = test_t
        #self.ohe_headers = ohe_headers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = args.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = args.lr_anneal_step, gamma = args.lr_anneal_rate)
        self.criterion = PinballLoss(self.args.training_tau, self.args.OUTPUT_SIZE, self.args.device)
        self.epochs = 0
        self.max_epochs = args.num_epochs
        #self.run_id = str(run_id)
        #self.prod_str = 'prod' if config['prod'] else 'dev'
        #self.log = Logger("../logs/train%s%s%s" % (self.config['variable'], self.prod_str, self.run_id))
        self.csv_save_path = 'model_checkpoints/esrnn/results/'

    def train_epochs(self):
        max_loss = 1e8
        start_time = time.time()
        for e in range(self.max_epochs):
            print('Training epoch -> {}'.format(e+1))
            epoch_loss = self.train() 
            if epoch_loss < max_loss:
                self.save()
            epoch_val_loss = self.val()
            if e == 0:
                file_path = os.path.join(self.csv_save_path, 'validation_losses.csv')
                with open(file_path, 'w') as f:
                    f.write('epoch,training_loss,validation_loss\n')
            with open(file_path, 'a') as f:
                f.write(','.join([str(e), str(epoch_loss), str(epoch_val_loss)]) + '\n')
        print('Total Training Mins: %5.2f' % ((time.time()-start_time)/60))

    def train(self):
        self.model.train()
        epoch_loss = 0
        
        #for batch_num, (train, val, test) in enumerate(self.dl):
        #    print("Train_batch: %d" % (batch_num + 1))
        loss = self.train_batch(self.train_t, self.val_t, self.test_t)
        epoch_loss += loss
            
        #epoch_loss = epoch_loss / (batch_num + 1)
        self.epochs += 1

        # LOG EPOCH LEVEL INFORMATION
        '''
        print('[TRAIN]  Epoch [%d/%d]   Loss: %.4f' % (
            self.epochs, self.max_epochs, epoch_loss))
        info = {'loss': epoch_loss}

        self.log_values(info)
        self.log_hists()
        '''
        return epoch_loss

    def train_batch(self, train_t, val_t, test_t):
        self.optimizer.zero_grad()
        network_pred, network_act, _, _, loss_mean_sq_log_diff_level = self.model(train_t, val_t, test_t)
        loss = self.criterion(network_pred, network_act)
        loss.backward()
        nn.utils.clip_grad_value_(self.model.parameters(), self.args.gradient_clipping)
        self.optimizer.step()
        self.scheduler.step()
        return float(loss)

    def val(self):
        self.model.eval()
        with torch.no_grad():
            acts = []
            preds = []
            #info_cats = []

            hold_out_loss = 0
           # for batch_num, (train, val, test, info_cat, idx) in enumerate(self.dl):
            _, _, (hold_out_pred, network_output_non_train), \
            (hold_out_act, hold_out_act_deseas_norm), _ = self.model(self.train_t, self.val_t, self.test_t)
                
            hold_out_loss += self.criterion(network_output_non_train.unsqueeze(0).float(), hold_out_act_deseas_norm.unsqueeze(0).float())
            acts.extend(hold_out_act.view(-1).cpu().detach().numpy())
            preds.extend(hold_out_pred.view(-1).cpu().detach().numpy())
                
            #info_cats.append(info_cat.cpu().detach().numpy())
            
            #hold_out_loss = hold_out_loss / (batch_num + 1)

            #info_cat_overall = np.concatenate(info_cats, axis=0)
            _hold_out_df = po.DataFrame({'acts': acts, 'preds': preds})
            
            #cats = [val for val in self.ohe_headers[info_cat_overall.argmax(axis=1)] for _ in range(self.config['output_size'])]
            #_hold_out_df['category'] = cats

            #overall_hold_out_df = copy.copy(_hold_out_df)
            #overall_hold_out_df['category'] = ['Overall' for _ in cats]

            #overall_hold_out_df = po.concat((_hold_out_df, overall_hold_out_df))
            #grouped_results = overall_hold_out_df.groupby(['category']).apply(lambda x: np_sMAPE(x.preds, x.acts, x.shape[0]))

            #results = grouped_results.to_dict()
            results = _hold_out_df
            results['hold_out_loss'] = float(hold_out_loss.detach().cpu())

            #self.log_values(results)

            #file_path = os.path.join('..', 'grouped_results', self.run_id, self.prod_str)
            #os.makedirs(file_path, exist_ok=True)

            #print(results)
            #grouped_path = os.path.join(file_path, 'grouped_results-{}.csv'.format(self.epochs))
            results_path = 'model_checkpoints/esrnn/results/results_{}_epochs.csv'.format(self.epochs)
            results.to_csv(results_path)
            #self.csv_save_path = file_path

        return hold_out_loss.detach().cpu().item()

    def save(self, save_dir='model_checkpoints/esrnn/'):
        print('Loss decreased, saving model!')
        file_path = save_dir #os.path.join(save_dir, 'models', self.run_id, self.prod_str)
        model_path = os.path.join(file_path, 'model-{}.pyt'.format(self.epochs))
        os.makedirs(file_path, exist_ok=True)
        os.makedirs(file_path+'results/', exist_ok=True)
        torch.save({'state_dict': self.model.state_dict()}, model_path)
    
    '''
    def log_values(self, info):

        # SCALAR
        for tag, value in info.items():
            self.log.log_scalar(tag, value, self.epochs + 1)

    def log_hists(self):
        # HISTS
        batch_params = dict()
        for tag, value in self.model.named_parameters():
            if value.grad is not None:
                if "init" in tag:
                    name, _ = tag.split(".")
                    if name not in batch_params.keys() or "%s/grad" % name not in batch_params.keys():
                        batch_params[name] = []
                        batch_params["%s/grad" % name] = []
                    batch_params[name].append(value.data.cpu().numpy())
                    batch_params["%s/grad" % name].append(value.grad.cpu().numpy())
                else:
                    tag = tag.replace('.', '/')
                    self.log.log_histogram(tag, value.data.cpu().numpy(), self.epochs + 1)
                    self.log.log_histogram(tag + '/grad', value.grad.data.cpu().numpy(), self.epochs + 1)
            else:
                print('Not printing %s because it\'s not updating' % tag)

        for tag, v in batch_params.items():
            vals = np.concatenate(np.array(v))
            self.log.log_histogram(tag, vals, self.epochs + 1)
    '''

