import torch
import torch.nn as nn
import torch.optim as optim

from model_zoo.vanillaWaveNet import vanillaWaveNet
from config import vanillaWaveNet_args
from utils.trainer import Trainer

torch.manual_seed(31415)

args = vanillaWaveNet_args()

model = vanillaWaveNet(args).to(args.device)

model

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

trainer = Trainer(model, criterion, optimizer, args)

trainer.train(print_losses=True)




