import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class BackscatterDNN(nn.Module):
    def __init__(self, input_size, numClasses, lr=0.001, optimizer='Adam', manual_seed=None, device='cpu'):
        super(BackscatterDNN, self).__init__()
        if manual_seed != None:
            T.manual_seed(manual_seed)
            print(f'---Manual seed of DNN:{manual_seed}---')
        self.lr = lr
        print(f'learning rate:{lr}')
       # Build a feed-forward network
        self.model = nn.Sequential(nn.Linear(input_size, 600),
                                   nn.Tanh(),
                                   nn.Linear(600,1000),
                                   nn.Tanh(),
                                   nn.Linear(1000,600),
                                   nn.Tanh(),
                                   nn.Linear(600, 1),
                                   nn.Sigmoid(),
                                  )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.BCELoss()
        
        self.device = T.device(device if T.cuda.is_available() else 'cpu')
        self.to(device)
        
    def forward(self, input_data):
        x =self.model(input_data)      
        return x


def accuracy(model, ds):
  # ds is a iterable Dataset of Tensors
    n_correct = 0; n_wrong = 0

    # alt: create DataLoader and then enumerate it
    for i in range(len(ds)):
        inpts = ds[i]['predictors']
        target = ds[i]['target']    # float32  [0.0] or [1.0]
        with T.no_grad():
            oupt = model(inpts)

        # avoid 'target == 1.0'
        if target < 0.5 and oupt < 0.5:  # .item() not needed
            n_correct += 1
        elif target >= 0.5 and oupt >= 0.5:
            n_correct += 1
        else:
            n_wrong += 1

    return (n_correct * 1.0) / (n_correct + n_wrong)

def acc_coarse(model, ds):
    inpts = ds[:]['predictors']  # all rows
    targets = ds[:]['target']    # all target 0s and 1s
    with T.no_grad():
        oupts = model(inpts)         # all computed ouputs
    pred_y = oupts >= 0.5        # tensor of 0s and 1s
    num_correct = T.sum(targets==pred_y)
    acc = (num_correct.item() * 1.0 / len(ds))  # scalar
    return acc