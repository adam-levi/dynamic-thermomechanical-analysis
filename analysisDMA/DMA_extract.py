import os
import pandas as pd
import codecs
import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO

#imports raw DMA data en masse and reformats and saves as csv so it is ready for directly importing into pandas dataframe structure
def data_to_df(file_name):
    f  = open(file_name, mode='r', encoding='utf-16').read().split('\n')  ##need to fix this, add with open and close()
    f = [i.split('\t') for i in f]
    df=pd.DataFrame(f)
    nsigs=len(df.columns)
    data_index = df[df[0]=='StartOfData'].index.values
    sig_index = df[df[0]=='Sig1'].index.values
    df.columns = df[1].iloc[sig_index[0] : sig_index[0] + nsigs].values.tolist()
    df = df.drop(range(0,data_index[0]+1)).reset_index()
    df = df.drop('index', axis = 1)
    df = df.dropna().astype(float)
    
 
    return df

#takes no arguments, this function just needs to be run the same folder as the raw data files to reformat them as .csv files ready for import to a pandas df
#Removes meta data from original file
def data_reformat():

    
    filenames = []
    
    for x in os.listdir():
        if x.endswith(".txt"):
            filenames.append(x)
            
    for i in filenames:
        
        file_name = i
        
        df = data_to_df(file_name)
        
        file_name = file_name.split('.')
        file_name = file_name[0] + '.csv'
    
        df.to_csv(file_name,index=None)
