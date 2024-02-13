import numpy as np
import os
import sys
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

from py_torch_qrnn_adapt.utils import create_folds, batches
#from utils import create_folds, batches
#from torch_utils import clip_gradient, logsumexp

from sklearn.metrics import mean_squared_error
from tqdm import tqdm


class QuantileNetworkMM(nn.Module):
    def __init__(self,n_in,n_out,y_dim):
        super(QuantileNetworkMM).__init__()
        self.n_in=n_in
        self.n_out=n_out
        self.y_dim=y_dim
        self.linear=nn.Sequential(
            nn.Linear(self.n_in,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,self.n_out)
        )

    def forward(self,x):
        out=self.linear(x)
        return out

class QuantileNetwork():
    
    def __intit__(self,quantiles,loss='quantile'):
        self.quantiles=quantiles
        self.lossfn=loss

    def predict(self, X):
        return self.model.predict(X)
    
    def PSNR(y_true,y_pred):
        mse = mean_squared_error(y_true,y_pred)
        maxval = np.amax(y_true)
        PSNR = 10*np.log10(maxval/mse)
        
        return PSNR
    
    def calc_outrate(y_test_np,preds):
        outcount = 0
        for i in range(np.shape(y_test_np)[0]):
            for j in range(np.shape(y_test_np)[1]):
                if y_test_np[i,j] < preds[i,j,0] or y_test_np[i,j] > preds[i,j,2]:
                    outcount = outcount +1

        outrate = outcount/np.size(y_test_np)
        return outrate

def fit_quantiles(X,y,X_val,y_val,quantiles,n_epochs,batch_size,loss='quantile'):

    X_mean=np.mean(X)
    y_mean=np.mean(y)
    n_in=len(X_mean)
    n_out=len(quantiles)
    y_dim=len(y_mean)

    #Initiate loss tracking
    train_losses=np.zeros(n_epochs)
    val_losses=np.zeros(n_epochs)

    model=QuantileNetworkMM(n_in,n_out,y_dim)

    optimizer = optim.SGD(model.parameters()) #Set optimiser, atm Stochastic Gradient Descent
    tquantiles = autograd.Variable(torch.FloatTensor(quantiles), requires_grad=False) # I dont know what this does yet
    
    train_indices=np.arange(X.shape[0], dtype=int)
    val_indices=np.arange(X_val.shape[0], dtype=int)


    # Univariate quantile loss
    def quantile_loss(yhat, idx):
        z = y[idx,None] - yhat
        return torch.max(tquantiles[None]*z, (tquantiles[None] - 1)*z)

    # Marginal quantile loss for multivariate response
    def marginal_loss(yhat, idx):
        z = y[idx,:,None] - yhat
        return torch.max(tquantiles[None,None]*z, (tquantiles[None,None] - 1)*z)
    
    if len(y.shape) == 1 or y.shape[1] == 1: #If Univariate
        lossfn = quantile_loss
    else:
        lossfn=marginal_loss

    for epoch in range(n_epochs):
        print('Epoch {}'.format(epoch+1))
        sys.stdout.flush()

        train_loss = torch.Tensor([0])

        for batch in tqdm(batches(train_indices, batch_size, shuffle=True), desc="Batch number: "):
            idx = autograd.Variable(torch.LongTensor(batch), requires_grad=False)
            
            model.train() #Initialise train mode
            model.zero_grad() #Reset gradient
            yhat = model(X[idx]) #Run algorithm

            loss=lossfn(yhat) #Run loss function
            loss.backward() #Calculate gradient

            optimizer.step()

            train_loss=train_loss+loss.data #Increment loss

        validation_loss = torch.Tensor([0])
        for batch_idx, batch in enumerate(batches(val_indices, batch_size, shuffle=False)):
            idx = autograd.Variable(torch.LongTensor(batch), requires_grad=False)

            model.eval() #Set evaluation mode
            model.zero_grad() #Reset gradient

            yhat=model(X[idx])

            validation_loss=validation_loss+lossfn(yhat, idx).sum()

    train_losses[epoch] = train_loss.data.numpy() / float(len(train_indices))
    val_losses[epoch] = validation_loss.data.numpy() / float(len(val_indices))

    return model

