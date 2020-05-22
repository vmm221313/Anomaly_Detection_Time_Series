import pandas as po
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from model_zoo.nbeats import NBeatsNet

device = torch.device('cuda:0' if torch.cuda. is_available() else 'cpu')

model = NBeatsNet(device, backcast_length=12*24*7, forecast_length=12*24)

df = po.read_csv('data/2017_energy_5min_noTransform.csv')

train = torch.tensor(df[:12*24*7].values).view(1, -1).float()

train.shape

backcast, forecast = model(train)

backcast.shape

forecast.shape

backcast

forecast


