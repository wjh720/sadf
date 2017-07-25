import time
import os
import sys
import logging
import numpy as np
import copy
import importlib
import librosa
import soundfile
import collections


from keras.models import Sequential, Model
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import Dense, Dropout, Flatten,Permute
from keras.layers import Conv1D, MaxPooling1D,Conv2D,MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Dense, LSTM, Lambda, Embedding, Reshape
from keras.layers.merge import Concatenate
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.constraints import max_norm
from keras.layers import LSTM,Add
from keras import backend as K
import tensorflow as tf
from keras.callbacks import ModelCheckpoint


class Learner():
    def __init__(self):
        pass

    def prepare(self):
        f = file('label.npy', 'r')
        self.label = np.load(f)
        f.close()

        f = file('data.npy', 'r')
        self.data = np.load(f)
        f.close()

        self.label = self.label.reshape(-1, 1)

        print(self.data.shape)
        print(self.label.shape)

    def learn(self):
        tbCallBack = keras.callbacks.TensorBoard(log_dir='../Graph', histogram_freq=0, write_graph=True, write_images=True)
        checkpointer = ModelCheckpoint(filepath='/data/tmpsrt1/log_new/wph', save_best_only=True, period = 10, \
                        verbose = 1,save_weights_only = True)
        
        print(' Begin fitting ')

        self.model.fit(
            x = self.data,
            y = self.label,
            batch_size = 256,
            epochs = 10000,
            validation_split = 0.2,
            verbose = 2,
            shuffle = True,
            callbacks = [tbCallBack,checkpointer]
        )

        print(' End fitting ')


    def create_model(self):
        pass


    def work(self):
        create_model()
        learn()

a = Learner()
a.work()


