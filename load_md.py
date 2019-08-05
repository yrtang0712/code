import pandas as pd
import numpy as np
import os

shape = (5120,4096)
data_path = '/home/data/remoteDir/server_196/mem_data/' 
X_data_path = '/data/HFAS_198'
Y_data_path = '/data/massDisk/mem_ret_data'
inx_path = os.path.join(data_path ,'.index/tick.csv')
col_path = os.path.join(data_path ,'.index/code.csv')

inx = pd.read_csv(inx_path).set_index('tick_time')['idx']
col = pd.read_csv(col_path,dtype={'stock_code':str}).set_index('stock_code')['idx']
                  
def get_mem_data_by_tick(day,tag,ticks=None,codes=None,dtype='float32',fillna=True,change_zero=True):
    year = str(day // 10000).zfill(4)
    month = str(day // 100 % 100).zfill(2)
    date= str(day% 100).zfill(2)    
    path = os.path.join(data_path,year,month,date,tag)
    assert os.path.exists(path),f'{path} not exists'
    
    ticks = inx.index.tolist() if ticks is None else ticks
    codes = col.index.tolist() if codes is None else codes
    
    r = inx.reindex(ticks).values
    c = col.reindex(codes).values
    
    file = np.memmap(path,mode='r',shape=shape,dtype=dtype)
    df = pd.DataFrame(file[r][:,c] ,index = ticks ,columns= codes)
    if change_zero:
        df = df.replace(0,np.nan)
    if fillna:
        df = df.fillna(method='ffill')
    return df
    


def get_X_data_by_tick(user,day,ticks=None,codes=None,dtype='float32',fillna=True,change_zero=True):
  
    path = os.path.join(Y_data_path,user,day)
    assert os.path.exists(path),f'{path} not exists'
    
    ticks = inx.index.tolist() if ticks is None else ticks
    codes = col.index.tolist() if codes is None else codes
    
    r = inx.reindex(ticks).values
    c = col.reindex(codes).values
    
    file = np.memmap(path,mode='r',shape=shape,dtype=dtype)
    df = pd.DataFrame(file[r][:,c] ,index = ticks ,columns= codes)
    if change_zero:
        df = df.replace(0,np.nan)
    if fillna:
        df = df.fillna(method='ffill')
    return df


def get_Y_data_by_tick(day,tag,ticks=None,codes=None,dtype='float32',fillna=True,change_zero=True):
  
    path = os.path.join(Y_data_path,day,tag)
    assert os.path.exists(path),f'{path} not exists'
    
    ticks = inx.index.tolist() if ticks is None else ticks
    codes = col.index.tolist() if codes is None else codes
    
    r = inx.reindex(ticks).values
    c = col.reindex(codes).values
    
    file = np.memmap(path,mode='r',shape=shape,dtype=dtype)
    df = pd.DataFrame(file[r][:,c] ,index = ticks ,columns= codes)
    if change_zero:
        df = df.replace(0,np.nan)
    if fillna:
        df = df.fillna(method='ffill')
    return df



def main():

    i = get_Y_data_by_tick("20160104","3min_ret",[93000000],["000001"])
    print(i)

if __name__ == "__main__":
    main()