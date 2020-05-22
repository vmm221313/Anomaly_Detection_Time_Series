import torch
import torch.nn as nn
from model_zoo.custom_layers.DRNN import DRNN


class ESRNN(nn.Module):
    def __init__(self, args):
        super(ESRNN, self).__init__()
        self.args = args
        self.add_nl_layer = True
        
        init_lev_sms = []
        init_seas_sms = []
        init_seasonalities = []

        init_lev_sms.append(nn.Parameter(torch.Tensor([0.5]), requires_grad=True))
        init_seas_sms.append(nn.Parameter(torch.Tensor([0.5]), requires_grad=True))    
        init_seasonalities.append(nn.Parameter((torch.ones(self.args.SEASONALITY)*0.5), requires_grad=True)) #since I am only considering daily seasonality for now

        self.init_lev_sms = nn.ParameterList(init_lev_sms)
        self.init_seas_sms = nn.ParameterList(init_seas_sms)
        self.init_seasonalities = nn.ParameterList(init_seasonalities)
        
        self.nl_layer = nn.Linear(self.args.STATE_H_SIZE, self.args.STATE_H_SIZE)
        self.act = nn.Tanh()
        self.scoring = nn.Linear(self.args.STATE_H_SIZE, self.args.OUTPUT_SIZE)
        self.logistic = nn.Sigmoid()
        
        self.resid_drnn = ResidualDRNN(self.args)
        
    def forward(self, train, val, test, testing = False):
        lev_sms = self.logistic(self.init_lev_sms[0].squeeze()) #alpha
        seas_sms = self.logistic(self.init_seas_sms[0].squeeze()) #gamma
        init_seasonalities = self.init_seasonalities[0].view(1, -1)

        seasonalities = []
        for i in range(self.args.SEASONALITY):
            seasonalities.append(torch.exp(init_seasonalities[:, i]))
        seasonalities.append(torch.exp(init_seasonalities[:, 0])) #why add it once again?
        
        if testing: 
            train = torch.cat((train, val), dim=1)
            
        train = train.float()
        
        levs = []
        log_diff_of_levels = []

        levs.append(train[:, 0]/seasonalities[0]) #the first level has to be defined seperately and subsequent levels are defined based on it (SEE HW)
        for i in range(1, train.shape[1]):
            new_lev = lev_sms*(train[:, i]/seasonalities[i]) + (1 - lev_sms)*levs[i-1] #to normalize
            levs.append(new_lev)
            log_diff_of_levels.append(torch.log(new_lev/levs[i - 1]))
            seasonalities.append(seas_sms*(train[:, i]/new_lev) + (1 - seas_sms)*seasonalities[i]) #why not seasonalities[i-m]?
            
            
        seasonalities_stacked = torch.stack(seasonalities).transpose(1, 0)
        levs_stacked = torch.stack(levs).transpose(1, 0)
        
        loss_mean_sq_log_diff_level = 0
        #level_variablility penalty not implemented
        
        if self.args.OUTPUT_SIZE > self.args.SEASONALITY:
            start_seasonality_ext = seasonalities_stacked.shape[1] - self.args.SEASONALITY
            end_seasonality_ext = start_seasonality_ext + self.args.OUTPUT_SIZE - self.args.SEASONALITY
            seasonalities_stacked = torch.cat((seasonalities_stacked, seasonalities_stacked[:, start_seasonality_ext:end_seasonality_ext]), dim=1)
            
        window_input_list = []
        window_output_list = []
        for i in range(self.args.INPUT_SIZE - 1, train.shape[1]):
            input_window_start = i + 1 - self.args.INPUT_SIZE
            input_window_end = i + 1

            train_deseas_window_input = train[:, input_window_start:input_window_end]/seasonalities_stacked[:, input_window_start:input_window_end]
            train_deseas_norm_window_input = (train_deseas_window_input/levs_stacked[:, i].unsqueeze(1))
            window_input_list.append(train_deseas_norm_window_input)

            output_window_start = i + 1
            output_window_end = i + 1 + self.args.OUTPUT_SIZE

            if i < train.shape[1] - self.args.OUTPUT_SIZE: #all other than right edge cases 
                train_deseas_window_output = train[:, output_window_start:output_window_end]/seasonalities_stacked[:, output_window_start:output_window_end]
                train_deseas_norm_window_output = (train_deseas_window_output/levs_stacked[:, i].unsqueeze(1))
                window_output_list.append(train_deseas_norm_window_output)

        window_input = torch.cat([i.unsqueeze(0) for i in window_input_list], dim=0) #list to tensor
        window_output = torch.cat([i.unsqueeze(0) for i in window_output_list], dim=0)

        self.train()
        #print(window_input.shape)
        network_pred = self.series_forward(window_input[:-self.args.OUTPUT_SIZE]) #your favorite check
        network_act = window_output
        
        self.eval() # validation
        network_output_non_train = self.series_forward(window_input) #output when given the full train seq
        
        #print(network_output_non_train[-1].shape)
        #print(seasonalities_stacked[:, -self.args.OUTPUT_SIZE:].shape)
        #here we are renormalizing/reseasoning (XD) future prediction using the seasonality values at the end of the train_seq
        hold_out_output_reseas = network_output_non_train[-1] * seasonalities_stacked[:, -self.args.OUTPUT_SIZE:] 
        hold_out_output_renorm = hold_out_output_reseas * levs_stacked[:, -1].unsqueeze(1)
        
        hold_out_pred = hold_out_output_renorm * torch.gt(hold_out_output_renorm, 0).float() #??
        hold_out_act = test if testing else val
        
        #print(hold_out_act.shape)
        #print(seasonalities_stacked[:, -self.args.OUTPUT_SIZE:].shape)
        hold_out_act_deseas = hold_out_act.float() / seasonalities_stacked[:, -self.args.OUTPUT_SIZE:]
        hold_out_act_deases_norm = hold_out_act_deseas / levs_stacked[:, -1].unsqueeze(1)
        
        self.train()
        
        return network_pred, network_act, (hold_out_pred, network_output_non_train), (hold_out_act, hold_out_act_deases_norm), loss_mean_sq_log_diff_level
        
        
    def series_forward(self, data):
        data = self.resid_drnn(data) 
        
        if self.add_nl_layer:
            data = self.nl_layer(data)
            data = self.act(data)
            
        data = self.scoring(data)
        return data


class ResidualDRNN(nn.Module):
    def __init__(self, args):
        super(ResidualDRNN, self).__init__()
        self.args = args
        
        layers = []
        for grp_num in range(len(self.args.DILATIONS)): 
            
            if grp_num == 0:
                input_size = self.args.INPUT_SIZE
            else:
                input_size = self.args.STATE_H_SIZE #for stacked lstms
                
            l = DRNN(input_size, self.args.STATE_H_SIZE, n_layers = len(self.args.DILATIONS[grp_num]), dilations = self.args.DILATIONS[grp_num], cell_type=self.args.RNN_CELL_TYPE)
            layers.append(l)
            
        self.rnn_stack = nn.Sequential(*layers)
        
    def forward(self, input_data):
        for layer_num in range(len(self.rnn_stack)):
            residual = input_data
            out, _ = self.rnn_stack[layer_num](input_data)
            if layer_num > 0:
                out += residual
                
            input_data = out
            
        return out


