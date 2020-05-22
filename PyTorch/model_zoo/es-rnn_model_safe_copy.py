# +
#let's do this layer by layer

# +
#first start with the deseas/denorm layer
# -

import torch
import torch.nn as nn

import pandas as po

SEASONALITY = 12*24
INPUT_SIZE = 12*24*7 #aka train_seq_len in other files
OUTPUT_SIZE = 12*24 #aka val_seq_len in other files

init_lev_sms = []
init_seas_sms = []
init_seasonalities = []

num_series = 1

init_lev_sms.append(nn.Parameter(torch.Tensor([0.5]), requires_grad=True))
init_seas_sms.append(nn.Parameter(torch.Tensor([0.5]), requires_grad=True))    
init_seasonalities.append(nn.Parameter((torch.ones(SEASONALITY)*0.5), requires_grad=True)) #since I a only considering daily seasonality for now

init_lev_sms = nn.ParameterList(init_lev_sms)
init_seas_sms = nn.ParameterList(init_seas_sms)
init_seasonalities = nn.ParameterList(init_seasonalities)

sigmoid = nn.Sigmoid()

# +
#since there's only one series in our dataset no need to batch series
# -

lev_sms = sigmoid(init_lev_sms[0].squeeze()) #alpha
seas_sms = sigmoid(init_seas_sms[0].squeeze()) #gamma
init_seasonalities = init_seasonalities[0].view(1, -1)

init_seasonalities.shape

seasonalities = []
for i in range(SEASONALITY):
    seasonalities.append(torch.exp(init_seasonalities[:, i]))

seasonalities.append(torch.exp(init_seasonalities[:, 0])) #why add it once again?

# +
#since we're no longer batching at this step, we cannot use the custom dataloader
# -

df = po.read_csv('../data/2017_energy_5min_noTransform.csv')
df.head()

# +
#in the M4 dataset train shape - num_series * num_datapoints (verified)
# -

train = torch.tensor(df.values[:12*24*250]).T
val = torch.tensor(df.values[12*24*250:12*24*300]).T
test = torch.tensor(df.values[12*24*300:12*24*350]).T

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

# +
#level_variablility penalty not implemented
# -

if OUTPUT_SIZE > SEASONALITY:
    start_seasonality_ext = seasonalities_stacked.shape[1] - SEASONALITY
    end_seasonality_ext = start_seasonality_ext + OUTPUT_SIZE - SEASONALITY
    seasonalities_stacked = torch.cat((seasonalities_stacked, seasonalities_stacked[:, start_seasonality_ext:end_seasonality_ext]), dim=1)

window_input_list = []
window_output_list = []

levs_stacked.shape

for i in range(INPUT_SIZE - 1, train.shape[1]):
    input_window_start = i + 1 - INPUT_SIZE
    input_window_end = i + 1
    
    train_deseas_window_input = train[:, input_window_start:input_window_end]/seasonalities_stacked[:, input_window_start:input_window_end]
    train_deseas_norm_window_input = (train_deseas_window_input/levs_stacked[:, i].unsqueeze(1))
    window_input_list.append(train_deseas_norm_window_input)
    
    output_window_start = i + 1
    output_window_end = i + 1 + OUTPUT_SIZE
    
    if i < train.shape[1] - OUTPUT_SIZE: #all other than right edge cases 
        train_deseas_window_output = train[:, output_window_start:output_window_end]/seasonalities_stacked[:, output_window_start:output_window_end]
        train_deseas_norm_window_output = (train_deseas_window_output/levs_stacked[:, i].unsqueeze(1))
        window_output_list.append(train_deseas_norm_window_output)
    
    window_input = torch.cat([i.unsqueeze(0) for i in window_input_list], dim=0) #list to tensor
    window_output = torch.cat([i.unsqueeze(0) for i in window_output_list], dim=0)
    
    <PRE-PROCESSING DONE>
    
    break

class
