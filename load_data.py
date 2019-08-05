import os
import pandas as pd
import numpy as np
import load_md as foo
import time
from multiprocessing import Pool

def task(nth, days, codes):
    print('Start process %d to read data.' % os.getpid())
    data = pd.DataFrame()
    for d in days:
        df = data_processing(d, codes)
        data = data.append(df)
    print(data.shape)
    data.to_csv('./data/temp/data_%02d.csv' % nth)

def dataset(start=20180101, end=20181231, parallel_lines=12):
    codes = pd.read_csv('./data/code.csv', index_col=0)
    codes.loc[:, 'code'] = codes['code'].apply(lambda x: str(x).zfill(6))
    codes = codes.code.values

    target = read_target(start, end)
    date = np.unique(target.date.values)
    target.to_csv('./data/target.csv')

    if date.shape[0] < parallel_lines:
        parallel_lines = date.shape[0]

    n = date.shape[0] // parallel_lines
    k = date.shape[0] % parallel_lines
    p = Pool(parallel_lines)
    for i in range(k):
        p.apply_async(task, args=(i, date[i * (n + 1):(i + 1) * (n + 1)], codes))
    date2 = date[k*(n+1):]
    for i in range(k, parallel_lines):
        p.apply_async(task, args=(i, date2[(i-k)*n:(i-k+1)*n], codes))
    p.close()
    p.join()

    def generate_dataset():
        data = pd.DataFrame()
        for j in range(0, parallel_lines):
            df = pd.read_csv('./data/temp/data_%02d.csv' % j, index_col=0)
            data = data.append(df)
        print(data.shape)
        data.to_csv('./data/data.csv')

    generate_dataset()
    print('Finished!')

    return date.shape[0]

def data_processing(date, codes=['000002']):

    keys = ['totbid', 'totoff', 'vol', 'last', 'low', 'high']
    keys.extend(['bid' + str(x) for x in range(1, 11)])
    keys.extend(['ask' + str(x) for x in range(1, 11)])
    keys.extend(['bid_vol' + str(x) for x in range(1, 11)])
    keys.extend(['ask_vol' + str(x) for x in range(1, 11)])

    data = pd.DataFrame()

    df = read_data(date, codes, keys)
    t1 = time.time()
    for c in codes:
        temp = pd.DataFrame(df[c].values)
        temp.columns = keys
        temp['code'] = c

        if temp.isnull().sum().max() >= 4802:
            continue

        temp['vol'] = temp['vol'].diff().fillna(0)

        temp['ask'] = temp[['ask' + str(x) for x in range(1, 11)]].mean(axis=1)
        temp['bid'] = temp[['bid' + str(x) for x in range(1, 11)]].mean(axis=1)
        temp.drop(['ask' + str(x) for x in range(1, 11)], axis=1, inplace=True)
        temp.drop(['bid' + str(x) for x in range(1, 11)], axis=1, inplace=True)

        temp['ask_vol'] = temp[['ask_vol' + str(x) for x in range(1, 11)]].mean(axis=1)
        temp['bid_vol'] = temp[['bid_vol' + str(x) for x in range(1, 11)]].mean(axis=1)
        temp.drop(['ask_vol' + str(x) for x in range(1, 11)], axis=1, inplace=True)
        temp.drop(['bid_vol' + str(x) for x in range(1, 11)], axis=1, inplace=True)

        m_lst = ['last', 'bid', 'ask']
        for i in m_lst:
            temp[i+'_mean'] = temp[i].ewm(span=4).mean()
            temp.drop(i, axis=1, inplace=True)

        data = data.append(temp[::100])
    print('Finished %d data processing, cost %.3fs' % (date, time.time() - t1))
    return data

def read_data(date, codes=['000002'], keys=None):
    fp64_keys = keys[:3]
    fp32_keys = keys[3:]

    t1 = time.time()
    data = pd.DataFrame()
    for i in fp64_keys:
        df = foo.get_mem_data_by_tick(date, i, codes=codes, dtype='float64').astype('float32')
        data = pd.concat([data, df], axis=1)

    for i in fp32_keys:
        df = foo.get_mem_data_by_tick(date, i, codes=codes, dtype='float32')
        data = pd.concat([data, df], axis=1)

    print('Finished %d, cost %.3fs' % (date, time.time() - t1))
    return data

def get_data(d=0):
    data = pd.read_csv('./data/data.csv', index_col=0).groupby('code')
    target = pd.read_csv('./data/target.csv').groupby('code')[:186]

    codes = pd.read_csv('./data/code.csv', index_col=0).code.values
    x_train, y_train, x_test, y_test = [], [], [], []
    
    for c in codes:
        try:
	        df = data.get_group(c)
	        obj = target.get_group(c)
        except KeyError:
            continue

        df.drop('code', axis=1, inplace=True)

        if df.shape[0] < 49*obj.shape[0]:
            continue

        df = df.fillna(method='bfill').fillna(method='ffill')
        df = df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
        for j in range(d+15, d+109):
            if j < d+105:
                x_train.append(df[(j-15)*49:j*49].values)
                y_train.append(obj['change'][j:j+1].values)
            elif j == d+105:
                pass
            else:
                x_test.append(df[(j-15)*49:j*49].values)
                y_test.append(obj['change'][j:j+1].values)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    y_train = y_train / 20 + 0.5
    y_train = np.clip(y_train, 0, 1)
    
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    y_test = y_test / 20 + 0.5
    
    return x_train, y_train, x_test, y_test

def read_target(start=20180101, end=20181231):
    df = pd.read_csv('./data/raw_data/target.csv', index_col=0)
    df.columns = ['date', 'code', 'change']

    target = pd.DataFrame()
    target = target.append(df[(start <= df.date) & (df.date <= end)])
    target['change'] = target['change'].apply(lambda x: x*100)

    return target

if __name__ == '__main__':
    start, end = 20180101, 20180131
    l = dataset(start, end)

    data = pd.read_csv('./data/data.csv', index_col=0).groupby('code')

    codes = pd.read_csv('./data/code.csv', index_col=0).code.values
    r_codes = []
    for c in codes:
        try:
	        df = data.get_group(c)
        except KeyError:
            continue

        if df.shape[0] < 49*l:
            continue
        r_codes.append(c)
        
    pd.DataFrame({'code': np.array(r_codes)}).to_csv('./data/r_code.csv')
