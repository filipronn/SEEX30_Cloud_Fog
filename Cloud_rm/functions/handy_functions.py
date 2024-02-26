import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import torch
import os

def normalise_input_df(df,labels):
    #Normalise to zero mean unit variance for all given column labels
    for i,col in enumerate(labels):
        tmp_mean=np.mean(df[col])
        tmp_var=np.var(df[col])

        df[col]=(df[col]-tmp_mean)/np.sqrt(tmp_var)
    
    return df

def add_noise(df,labels,sigma=0.001):
    for i,col in enumerate(labels):
        noise=np.random.normal(0,sigma,len(df[col]))
        df[col]=df[col]+noise
    return df

def save_model_and_test_data(filepath,model,X_test,y_test,pytorch=True,save_history=False,history_df=None):
    
    if pytorch:
        os.makedirs(filepath,exist_ok=True)
        torch.save(model,filepath+'/model_file')
    else:
        model.save(filepath=filepath)

    X_test.to_csv(filepath+'/xtest.csv',index=True)
    y_test.to_csv(filepath+'/ytest.csv',index=True)

    if save_history:
        history_df.to_csv(filepath+'/history.csv',index=False)

def load_model_and_test_data(filepath,pytorch=True,load_history=False):
    if pytorch:
        model=torch.load(filepath+'/model_file')
    else:
        model=tf.keras.models.load_model(filepath)
    X_test=pd.read_csv(filepath+'/xtest.csv',index_col=0)
    y_test=pd.read_csv(filepath+'/ytest.csv',index_col=0)
    
    if load_history:
        history_df=pd.read_csv(filepath+'/history.csv')
        return model, X_test, y_test, history_df
    else:
        return model, X_test, y_test
         
    

def dumb_down_surface(df):

    df['Surface_Desc_Dumb']=df['Surface_Desc'].str.split('-').str[0]
    return df

def split_data(X,y,split): #Train test split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1-split[0])
    res=1-split[0]
    X_test, X_val, y_test, y_val = train_test_split(X_test,y_test, test_size=res-split[1])

    return X_train, y_train, X_val, y_val, X_test, y_test


def add_MSI_noise(df,x_labels):
    #The two values of 10 in SNR is the channels that are not specified on the MSI instrument document.
    #SNR_from_channel_2=[102, 79, 45, 45, 34, 26, 20, 16, 10, 10, 2.8, 2.2]

    SNR_from_channel_2=[154, 168, 142, 117, 89, 105, 20, 174, 114, 50, 100, 100] #From https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/resolutions/spectral

    for i,label in enumerate(x_labels):
        col=df[label].to_numpy()
        noise_std=np.sqrt((np.mean(col**2))/SNR_from_channel_2[i])
        print("Noise standard deviation for "+str(label)+": "+str(noise_std))
        noise=np.random.normal(0,noise_std,size=len(col)) #Zero mean
        df[label]=col+noise

    return df
