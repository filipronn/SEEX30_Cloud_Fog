import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.callbacks import Callback
from sklearn.model_selection import train_test_split

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


