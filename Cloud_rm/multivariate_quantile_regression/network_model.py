import numpy as np
import os
import sys
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

from multivariate_quantile_regression.utils import batches
#from utils import create_folds, batches
#from torch_utils import clip_gradient, logsumexp

from sklearn.metrics import mean_squared_error
from tqdm import tqdm


class QuantileNetworkMM(nn.Module):
    def __init__(self,n_in,n_out,y_dim, X_means, X_stds, y_mean, y_std):
        super(QuantileNetworkMM, self).__init__()
        self.X_means = X_means
        self.X_stds = X_stds
        self.y_mean = y_mean
        self.y_std = y_std
        self.n_in=n_in
        self.n_out=n_out
        self.y_dim=y_dim
        self.linear=nn.Sequential(
            nn.Linear(self.n_in,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64, self.n_out if self.y_dim == 1 else self.n_out * self.y_dim)
        )
        self.softplus = nn.Softplus()

    def forward(self,x):
        fout = self.linear(x)
        # If we are dealing with multivariate responses, reshape to the (d x q) dimensions
        if len(self.y_mean.shape) != 1:
            fout = fout.reshape((-1, self.y_mean.shape[1], self.n_out))

        # If we only have 1 quantile, no need to do anything else
        if self.n_out == 1:
            return fout

        # Enforce monotonicity of the quantiles
        return torch.cat((fout[...,0:1], fout[...,0:1] + torch.cumsum(self.softplus(fout[...,1:]), dim=-1)), dim=-1)
    
    def predict(self,x):
        self.eval()
        self.zero_grad()
        tX=torch.FloatTensor(x)
        out=self.forward(tX)
        return out.data.numpy()

class QuantileNetwork():
    def __init__(self,quantiles,loss='quantile'):
        self.quantiles=quantiles
        self.lossfn=loss

    def fit(self, X, y, train_indices, validation_indices, batch_size, nepochs):
        self.model = fit_quantiles(X, y, train_indices, validation_indices, quantiles=self.quantiles, batch_size=batch_size, n_epochs=nepochs)

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

def fit_quantiles(X,y,train_indices,validation_indices,quantiles,n_epochs,batch_size,loss='quantile',file_checkpoints=True):

    X_mean=np.mean(X,axis=0,keepdims=True)
    X_std=np.std(X,axis=0,keepdims=True)
    y_mean=np.mean(y,axis=0,keepdims=True)
    y_std=np.std(y,axis=0,keepdims=True)
    n_in=len(X_mean[0])
    n_out=len(quantiles)
    y_dim=len(y_mean[0])

    tX = torch.FloatTensor(X)
    tY = torch.FloatTensor(y)

    #Initiate loss tracking
    train_losses=np.zeros(n_epochs)
    val_losses=np.zeros(n_epochs)
    val_losses[0]=10000000 #For finding lowest new validation error later

    model=QuantileNetworkMM(n_in,n_out,y_dim,X_mean,X_std,y_mean,y_std)

    optimizer = optim.Adam(model.parameters()) #Set optimiser, atm Stochastic Gradient Descent
    tquantiles = torch.FloatTensor(quantiles)
    
    train_indices=np.sort(train_indices)
    val_indices=np.sort(validation_indices)


    # Univariate quantile loss
    def quantile_loss(yhat, idx):
        z = tY[idx,None] - yhat
        return torch.max(tquantiles[None]*z, (tquantiles[None] - 1)*z)

    # Marginal quantile loss for multivariate response
    def marginal_loss(yhat, idx):
        z = tY[idx,:,None] - yhat
        return torch.max(tquantiles[None,None]*z, (tquantiles[None,None] - 1)*z)
    
    if len(y.shape) == 1 or y.shape[1] == 1: #If Univariate
        lossfn = quantile_loss
    else:
        lossfn = marginal_loss

    for epoch in range(n_epochs):
        print('Epoch {}'.format(epoch+1))
        sys.stdout.flush()

        train_loss = torch.Tensor([0])
        
        for batch in tqdm(batches(train_indices, batch_size, shuffle=True),
                          total=int(np.ceil(len(train_indices)/batch_size)),
                          desc="Batch number"):
            idx = torch.LongTensor(batch)

            model.train() #Initialise train mode
            model.zero_grad() #Reset gradient

            yhat = model(tX[idx]) #Run algorithm

            loss=lossfn(yhat,idx).sum() #Run loss function
            loss.backward() #Calculate gradient

            optimizer.step()

            train_loss=train_loss+loss.data #Increment loss

        validation_loss = torch.Tensor([0])
        for batch_idx, batch in enumerate(batches(val_indices, batch_size, shuffle=False)):
            idx = torch.LongTensor(batch)

            model.eval() #Set evaluation mode
            model.zero_grad() #Reset gradient

            yhat=model(tX[idx])

            validation_loss=validation_loss+lossfn(yhat, idx).sum()

        print('Training loss {}'.format(train_loss.data.numpy()/float(len(train_indices)))+
              ' Validation loss {}'.format(validation_loss.data.numpy()/float(len(val_indices))),
              end=None)
        sys.stdout.flush()

        train_loss = train_loss.data.numpy() / float(len(train_indices))
        validation_loss = validation_loss.data.numpy() / float(len(val_indices))

        if validation_loss[0]<np.min(val_losses[val_losses!=0.0]):
            if file_checkpoints:
                torch.save(model,'tmp_file')            
            print("----New best validation loss---- {}".format(validation_loss))

        train_losses[epoch] = train_loss
        val_losses[epoch] = validation_loss

    if file_checkpoints:
        model=torch.load('tmp_file')
        os.remove('tmp_file')
        print("Best model out of total max epochs found at epoch {}".format(np.argmin(val_losses)+1))


    return model

