import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

import tensorflow as tf
from tensorflow.python.keras import layers
import tensorflow_probability as tfp
from tensorflow.python.keras.callbacks import Callback
from sklearn.model_selection import train_test_split

import functions.handy_functions as hf

class LossHistory(Callback):
    def __init__(self):
        self.hist=[]

    def on_epoch_end(self, epoch, logs=None):
        epoch_log = {
            'epoch': epoch + 1,
            'training_loss': logs['loss'],
            'validation_loss': logs['val_loss']
        }
        self.hist.append(epoch_log)


def train_simple_model(df,x_labels,y_labels,split,epochs,batch_size):

    ##Split data##
    X=df[x_labels]
    y=df[y_labels]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=split[1])
    X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size=split[2])
    
    ##Create model##
    model=tf.keras.Sequential([
        layers.Dense(32,input_dim=len(x_labels),activation='linear'),
        layers.Dense(128,activation='relu'),
        #layers.Dense(128,activation='relu'),
        layers.Dense(len(y_labels),activation='linear')
    ])

    ##Compile model##
    model.compile(
        optimizer="adam",
        loss='mse',
        metrics=["mse"],
        run_eagerly=True
    )

    ##Train model##
    history=LossHistory()
    model.fit(X_train,y_train,epochs=epochs,validation_data=(X_val,y_val),batch_size=batch_size,callbacks=[history])

    #Make training history to a dataframe
    history_df=pd.DataFrame(history.hist,index=None)
    
    return model, history_df, X_test, y_test

def train_5layer_64neurons_model(df,x_labels,y_labels,split,epochs,batch_size):

    ##Split data##
    X=df[x_labels]
    y=df[y_labels]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=split[1])
    X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size=split[2])

    
    ##Create model##
    model=tf.keras.Sequential([
        layers.Dense(32,input_dim=len(x_labels),activation='linear'),
        layers.Dense(64,activation='relu'),
        layers.Dense(64,activation='relu'),
        layers.Dense(64,activation='relu'),
        layers.Dense(len(y_labels),activation='linear')
    ])

    ##Compile model##
    model.compile(
        optimizer="adam",
        loss='mse',
        metrics=["mse"],
        run_eagerly=True
    )

    ##Train model##
    history=LossHistory()
    model.fit(X_train,y_train,epochs=epochs,validation_data=(X_val,y_val),batch_size=batch_size,callbacks=[history])

    #Make training history to a dataframe
    history_df=pd.DataFrame(history.hist,index=None)
    
    return model, history_df, X_test, y_test


## Bayesian models ##

# kernel_size: the number of parameters in the dense weight matrix
def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    # Independent Normal Distribution
    return lambda t: tfp.distributions.Independent(tfp.distributions.Normal(loc=tf.zeros(n, dtype=dtype),
                                                scale=1),
                                     reinterpreted_batch_ndims=1)

def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(tfp.layers.IndependentNormal.params_size(n), dtype=dtype),
        tfp.layers.IndependentNormal(n)
    ])


def train_first_bayesian_model(df,x_labels,y_labels,split,epochs,batch_size):

    ##Split data##
    X=df[x_labels]
    y=df[y_labels]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=split[1])
    X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size=split[2])

    #kl_weight = 1.0 / batch_size
    #prior_params = {
    #    'prior_sigma_1': 1.5, 
    #    'prior_sigma_2': 0.1, 
    #    'prior_pi': 0.5 
    #}

    ##Create model##
    model=tf.keras.Sequential([
        layers.Dense(32,input_dim=len(x_labels),activation='linear'),
        layers.Dense(64,activation='relu'),
        layers.Dense(64,activation='relu'),
        tfp.layers.DenseVariational(len(y_labels),posterior,prior,activation='linear')
    ])

    ##Compile model##
    model.compile(
        optimizer="adam",
        loss='mse',
        metrics=["mse"],
        run_eagerly=True
    )

    ##Train model##
    history=LossHistory()
    model.fit(X_train,y_train,epochs=epochs,validation_data=(X_val,y_val),batch_size=batch_size,callbacks=[history])

    #Make training history to a dataframe
    history_df=pd.DataFrame(history.hist,index=None)
    
    return model, history_df, X_test, y_test



