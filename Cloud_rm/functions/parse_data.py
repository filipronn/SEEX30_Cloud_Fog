import pandas as pd

def parse(file_name):
    col_names=['Cloud_B01','Cloud_B02','Cloud_B03','Cloud_B04','Cloud_B05','Cloud_B06',
           'Cloud_B07','Cloud_B08','Cloud_B09','Cloud_B10','Cloud_B11','Cloud_B12','Cloud_B13',
           'Clear_B01','Clear_B02','Clear_B03','Clear_B04','Clear_B05','Clear_B06',
           'Clear_B07','Clear_B08','Clear_B09','Clear_B10','Clear_B11','Clear_B12','Clear_B13',
           'Sat_Zenith_Angle','Sun_Zenith_Angle','Azimuth_Diff_Angle','COT','Cloud_Type','Profile_ID','GOT','Water_Vapor','Surface_Desc']

    data=pd.read_csv(file_name,skiprows=53,sep=' ',skipinitialspace=True,header=None)
    m=dict(zip(range(1,37),col_names))
    data=data.rename(columns=m)
    data=data.drop(columns=[0])
    return data

