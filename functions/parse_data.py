import pandas as pd
import os

def parse(file_name):
    col_names=['Cloud_B01','Cloud_B02','Cloud_B03','Cloud_B04','Cloud_B05','Cloud_B06',
           'Cloud_B07','Cloud_B08','Cloud_B08A','Cloud_B09','Cloud_B10','Cloud_B11','Cloud_B12',
           'Clear_B01','Clear_B02','Clear_B03','Clear_B04','Clear_B05','Clear_B06',
           'Clear_B07','Clear_B08','Clear_B08A','Clear_B09','Clear_B10','Clear_B11','Clear_B12',
           'Sat_Zenith_Angle','Sun_Zenith_Angle','Azimuth_Diff_Angle','COT','Cloud_Type','Profile_ID','GOT','Water_Vapor','Surface_Desc']

    data=pd.read_csv(file_name,skiprows=53,sep=' ',skipinitialspace=True,header=None)
    m=dict(zip(range(1,37),col_names))
    data=data.rename(columns=m)
    data=data.drop(columns=[0])
    return data

def synth_dataloader(path_name='SMHIdata',drop_cols = True):
       #Set path_name to path containing data sets, ensure file names are as below
       #load all data
       data_water=parse(os.path.join(path_name, 'cloudrm2_water.dat'))
       data_clear=parse(os.path.join(path_name, 'cloudrm2_clear.dat'))
       data_ice=parse(os.path.join(path_name, 'cloudrm2_ice.dat'))
       data_mixed=parse(os.path.join(path_name, 'cloudrm2_mixed.dat'))

       #Concatenate all datasets, drop unnecessary cols and reset index
       df=pd.concat([data_water,data_clear,data_ice,data_mixed])
       if drop_cols:
              df=df.drop(columns=['Surface_Desc','Cloud_B01','Clear_B01'])
       df=df.reset_index(drop=True)

       return df
