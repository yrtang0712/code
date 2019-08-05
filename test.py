import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6, 7"

from model import LSTM_Model
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
ktf.set_session(session)

model = LSTM_Model(gpus=4)

for i in range(0, 76, 3):
    model.rolling(i)

model.save_model()