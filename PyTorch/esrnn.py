import torch
import torch.nn as nn
#import torch.optim as optim


#import time
import pandas as po
#from tqdm import tqdm
import matplotlib.pyplot as plt

from model_zoo.esrnn_model import ESRNN
from utils.esrnn_trainer import ESRNNTrainer

torch.manual_seed(31415)


class esrnn_args():
    SEASONALITY = 12*24
    INPUT_SIZE = 12*24*7 #aka train_seq_len in other files
    OUTPUT_SIZE = 12*24 #aka val_seq_len in other files
    STATE_H_SIZE = 100
    DILATIONS = ((1, 12*24), (3*12*24, 5*12*24))
    RNN_CELL_TYPE = 'LSTM'
    device = torch.device('cuda:0' if torch.cuda. is_available() else 'cpu')
    learning_rate = 0.001
    lr_anneal_rate = 0.5
    lr_anneal_step = 5
    percentile = 50
    training_percentile = 45
    tau = percentile/100
    training_tau = training_percentile/100
    num_epochs = 15


args = esrnn_args()

df = po.read_csv('data/2017_energy_5min_noTransform.csv')

# +
#for the split note that the way validation takes place is that the model is fed the last window of the train set and the output is compared
#against the actual val set. Hence the val set needs to be of length OUTPUT_SIZE. Consider changing this by changing how validation takes place
# -

train = torch.tensor(df.values[:12*24*250]).T
val = torch.tensor(df.values[12*24*250:12*24*250 + args.OUTPUT_SIZE]).T
test = torch.tensor(df.values[12*24*250 + args.OUTPUT_SIZE:]).T

model = ESRNN(args)

trainer = ESRNNTrainer(args, model, train, val, test)

# +
#maybe there's sth wrong with your call of DRNN
# -

trainer.train_epochs()

# +
#in the M4 dataset train shape - num_series * num_datapoints (verified)
# -

results = po.read_csv('model_checkpoints/esrnn/results/results_100_epochs.csv')

results.head()

plt.figure(figsize=(40, 20))
plt.plot(results['acts'], color = 'blue')
plt.plot(results['preds'], color = 'red')




