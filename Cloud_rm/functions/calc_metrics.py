from sklearn.metrics import mean_squared_error
import numpy as np

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