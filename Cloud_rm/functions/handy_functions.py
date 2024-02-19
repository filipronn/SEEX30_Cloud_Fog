import pandas as pd
import numpy as np

def normalise_input_df(df,labels):
    #Normalise to zero mean unit variance for all given column labels
    for i,col in enumerate(labels):
        tmp_mean=np.mean(df[col])
        tmp_var=np.var(df[col])

        df[col]=(df[col]-tmp_mean)/np.sqrt(tmp_var)
    
    return df