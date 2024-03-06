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
from sklearn.metrics import r2_score
from tqdm import tqdm

import matplotlib.pyplot as plt
from IPython.display import display, clear_output


class QuantileNetworkMM(nn.Module):
    def __init__(self,n_in,n_out,y_dim, X_means, X_stds, y_mean, y_std, seq, device, data_norm):
        super(QuantileNetworkMM, self).__init__()
        self.X_means = X_means
        self.X_stds = X_stds
        self.y_mean = y_mean
        self.y_std = y_std
        self.n_in=n_in
        self.n_out=n_out
        self.y_dim=y_dim
        self.linear=seq
        self.device=device
        self.data_norm=data_norm

        self.softplus = nn.Softplus()

    def forward(self,x):
        fout = self.linear(x)
        # If we are dealing with multivariate responses, reshape to the (d x q) dimensions
        if len(self.y_mean.shape) != 1:
            fout = fout.reshape((-1, self.y_mean.shape[1], self.n_out))

        # If we only have 1 quantile, no need to do anything else
        if self.n_out == 1:
            return fout

        monotonicity=True
        if monotonicity:
        # Enforce monotonicity of the quantiles
            return torch.cat((fout[...,0:1], fout[...,0:1] + torch.cumsum(self.softplus(fout[...,1:]), dim=-1)), dim=-1)
        else:
            return fout
    
    def predict(self,x):
        self.eval()
        self.zero_grad()
        if self.data_norm:
            tX=torch.tensor((x - self.X_means) / self.X_stds,dtype=torch.float,device=self.device)
            norm_out = self.forward(tX)
            out = norm_out.data.cpu() * self.y_std[...,None] + self.y_mean[...,None]
            return out.numpy()
        else:
            tX=torch.tensor(x,dtype=torch.float,device=self.device)
            out=self.forward(tX)
            return out.data.cpu().numpy()
        

        

class QuantileNetwork():
    def __init__(self,quantiles,loss='quantile'):
        self.quantiles=quantiles
        self.lossfn=loss
        if torch.cuda.is_available():
            self.device=torch.device('cuda')
        else:
            self.device=torch.device('cpu')

    def fit(self, X, y, train_indices, validation_indices, batch_size, nepochs, sequence,lr=0.001,data_norm=False,verbose=True,plot_training=False,early_break=False):
        self.model,self.train_loss,self.val_loss = fit_quantiles(X, y, train_indices, validation_indices,
                                                                  quantiles=self.quantiles, batch_size=batch_size, 
                                                                  sequence=sequence, n_epochs=nepochs,
                                                                  device=self.device,lr=lr,data_norm=data_norm,
                                                                  verbose=verbose,
                                                                  plot_training=plot_training,
                                                                  early_break=early_break)

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
    
    def quant_rate(y_true,y_pred):
        try:
            n_quants=np.shape(y_pred)[2]

        except:
            n_quants=np.shape(y_pred)[1]
        
        quantcount = np.zeros(n_quants)
        if len(np.shape(y_pred))==3: #If multiple output features
            for i in range(np.shape(y_pred)[0]):
                for j in range(np.shape(y_pred)[1]):
                    for k in range(n_quants):
                        if y_true[i,j] < y_pred[i,j,k]:
                            quantcount[k] = quantcount[k] + 1
        else: #If 1 output feature
            for i in range(np.shape(y_pred)[0]):
                for k in range(n_quants):
                    if y_true[i] < y_pred[i,k]:
                        quantcount[k] = quantcount[k] + 1


        quantrate = quantcount/np.size(y_true)
        return quantrate
    
    # Marginal quantile loss for multivariate response
    def mean_marginal_loss(y_true,y_pred,quantiles):
        z = y_true[:,:,None] - y_pred
        loss = np.sum(np.maximum(quantiles[None,None]*z, (quantiles[None,None] - 1)*z))
        return loss/(np.shape(y_true)[0])
    
    def quant_cross(y_pred):
        if len(np.shape(y_pred)) == 3:
            crosscount = 0
            for i in range(np.shape(y_pred)[0]):
                for j in range(np.shape(y_pred)[1]):
                    for k in range(np.shape(y_pred)[2]-1):
                        if y_pred[i,j,k+1] < y_pred[i,j,k]:
                            crosscount = crosscount + 1 

        else:
            crosscount = 0
            for i in range(np.shape(y_pred)[0]):
                for j in range(np.shape(y_pred)[1]-1):
                    if y_pred[i,j+1] < y_pred[i,j]:
                        crosscount = crosscount + 1 

        crossrate = crosscount/(np.size(y_pred)-np.size(y_pred[...,0]))

        return crossrate
    


def fit_quantiles(X,y,train_indices,validation_indices,quantiles,n_epochs,batch_size,sequence,lr,data_norm,early_break,
                  file_checkpoints=True,device=torch.device('cuda'),verbose=True,plot_training=False):


    X_mean=np.mean(X,axis=0,keepdims=True)
    X_std=np.std(X,axis=0,keepdims=True)
    y_mean=np.mean(y,axis=0,keepdims=True)
    y_std=np.std(y,axis=0,keepdims=True)
    try: #Only 1 input feature, catch
        n_in=len(X_mean[0])
    except:
        n_in=1
    n_out=len(quantiles)
    try:#Only 1 output feature, catch
        y_dim=len(y_mean[0])
    except:
        y_dim=1


    tquantiles = torch.tensor(quantiles,dtype=torch.float,device=device)

    #Initiate loss tracking
    train_losses=torch.zeros(n_epochs,device=device)
    val_losses=torch.zeros(n_epochs,device=device)
    val_losses[0]=10000000 #For finding lowest new validation error later

    #Initiate training/validation plot
    if plot_training:
        plt.ion()
        fig, ax=plt.subplots()
        ax.legend(['Training Loss','Validation Loss'])

    model=QuantileNetworkMM(n_in,n_out,y_dim,X_mean,X_std,y_mean,y_std,seq=sequence,device=device,data_norm=data_norm)

    optimizer = optim.Adam(model.parameters(),lr=lr) #Set optimiser, atm Stochastic Gradient Descent
    
    if early_break:
        no_improv_ctr = 0
        eps=1e-8

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
    
    def plot_loss():

        ax.cla()
        ax.plot(train_losses[:epoch].cpu().numpy())
        ax.plot(val_losses[:epoch].cpu().numpy())
        ax.legend(['Training Loss','Validation Loss'])
        ax.set_xlim((0,n_epochs))
        display(fig)
        clear_output(wait=True)
        
        #plt.show()
    
    if len(y.shape) == 1 or y.shape[1] == 1: #If Univariate
        lossfn = quantile_loss
    else:
        lossfn = marginal_loss

    X=torch.tensor(X,dtype=torch.float,device=device)    

    for epoch in range(n_epochs):
        if verbose:
            print('Epoch {}'.format(epoch+1))
            sys.stdout.flush()
        
        noise=torch.randn(X.shape,device=device)
        means=torch.mean(X,dim=0)*0.03
        noise=noise*means.repeat((len(noise[:,0]),1))
        tX=X+noise
        if data_norm:
            X_mean=torch.mean(tX,dim=0,keepdim=True)
            X_std=torch.std(tX,dim=0,keepdim=True)
            tX = torch.tensor((tX - X_mean) / X_std,dtype=torch.float,device=device)
            tY = torch.tensor((y - y_mean) / y_std,dtype=torch.float,device=device)
        else:
            tX = torch.tensor(tX,dtype=torch.float,device=device)
            tY = torch.tensor(y,dtype=torch.float,device=device)


        train_loss = torch.tensor([0],dtype=torch.float,device=device)
        if verbose:
            for batch in tqdm(batches(train_indices, batch_size, shuffle=True),
                            total=int(np.ceil(len(train_indices)/batch_size)),
                            desc="Batch number"):
                

                idx = torch.tensor(batch,dtype=torch.int64,device=device)

                model.train() #Initialise train mode
                model.zero_grad() #Reset gradient

                yhat = model(tX[idx]) #Run algorithm

                loss=lossfn(yhat,idx).sum() #Run loss function
                loss.backward() #Calculate gradient

                optimizer.step()

                train_loss=train_loss+loss.data #Increment loss
        else:
            for batch in batches(train_indices, batch_size, shuffle=True):

                idx = torch.tensor(batch,dtype=torch.int64,device=device)

                model.train() #Initialise train mode
                model.zero_grad() #Reset gradient

                yhat = model(tX[idx]) #Run algorithm

                loss=lossfn(yhat,idx).sum() #Run loss function
                loss.backward() #Calculate gradient

                optimizer.step()

                train_loss=train_loss+loss.data #Increment loss


        validation_loss = torch.tensor([0],dtype=torch.float,device=device)
        
        for batch_idx, batch in enumerate(batches(val_indices, batch_size, shuffle=False)):


            idx = torch.tensor(batch,dtype=torch.int64,device=device)

            model.eval() #Set evaluation mode
            model.zero_grad() #Reset gradient

            yhat=model(tX[idx])

            validation_loss=validation_loss+lossfn(yhat, idx).sum()

        if verbose:
            print('Training loss {}'.format((train_loss.data/float(len(train_indices))).data.cpu().numpy())+
                ' Validation loss {}'.format((validation_loss.data/float(len(val_indices))).data.cpu().numpy()),
                end=None)
            sys.stdout.flush()

        train_loss = (train_loss.data/float(len(train_indices)))
        validation_loss = (validation_loss.data/float(len(val_indices)))

        if validation_loss[0]<torch.min(val_losses[val_losses!=0.0]):
            if file_checkpoints:
                torch.save(model,'tmp_file')
            if verbose:
                print("----New best validation loss---- {}".format(validation_loss.data.cpu().numpy()))

            if early_break and torch.min(val_losses[val_losses!=0.0])-validation_loss[0] > eps:
                no_improv_ctr = 0

        elif early_break:
            no_improv_ctr += 1
            if no_improv_ctr == 100:
                print("---No improvement in 100 epochs, broke early---")
                break

        train_losses[epoch] = train_loss
        val_losses[epoch] = validation_loss

        if plot_training:
            plot_loss()

    if file_checkpoints:
        model=torch.load('tmp_file')
        os.remove('tmp_file')
        print("Best model out of total max epochs found at epoch {}".format(np.argmin(val_losses.data.cpu().numpy())+1))

    return model, train_losses, val_losses

