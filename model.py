from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import multi_gpu_model
import tensorflow as tf

from load_data import get_data
import pandas as pd
import numpy as np


class LSTM_Model:
    def __init__(self, gpus=1):
        with tf.device('/cpu:0'):
            model = Sequential()
            model.add(LSTM(100, return_sequences=True, input_shape=(735, 10)))
            model.add(LSTM(100))
            model.add(Dense(1))

        if gpus > 1:
            self.model = multi_gpu_model(model, gpus)
        else:
            self.model = model
        self.model.compile(loss='mse', optimizer=Adam(1e-4))
        self.p = []
        self.code = pd.read_csv('./data/r_code.csv', index_col=0).code.values
        self.date = np.unique(pd.read_csv('./data/target.csv').date.values)[106:]
    
    def rolling(self, i=0, batch_size=2048):
        x_train, y_train, x_test, y_test = get_data(i)
        early_stopping = EarlyStopping('loss', 0.0001, 5)
        self.model.fit(x_train, y_train, batch_size=2048, epochs=100, callbacks=[early_stopping])

        y_pred = self.model.predict(x_test, batch_size=500)
        r = pd.DataFrame({'change': y_test.flatten(), 'pred': y_pred.flatten()})

        save_result(i, y_pred, r)
    
    def save_result(self, i, y_pred, r):
        d = y_pred.shape[0] / self.code.shape[0]

        for j in range(d):
            df = pd.DataFrame({'code': self.code,'predict': y_pred[j::d].flatten()})
            df.loc[:, 'code'] = df['code'].apply(lambda x: str(x).zfill(6))
            df = df.sort_values('predict', ascending=False)
            df.to_csv('./data/csv/%d.csv' % self.date[i + j], header=None, index=0)

            self.p.append(r[j::d].corr().values[0, 1])

        df = pd.DataFrame({'p': np.array(self.p)})
        df.to_csv('result.csv')

    def save_model(self):
        self.model.save('./model/model.h5')
        