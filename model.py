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
import keras

num_repeat = 19

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

        self.label = self.label.repeat(num_repeat).reshape(-1, 1)

        n = self.data.shape[0]

        pdata = []
        for i in range(n):
            asd = self.data[i]
            #print(asd.shape)
            norm_asd = asd - np.mean(asd, axis = 0)
            #print(np.mean(norm_asd, axis = 0))
            norm_asd = norm_asd / np.std(norm_asd, axis = 0)
            #print(np.std(norm_asd, axis = 0))

            #time.sleep(30)

            for j in range(num_repeat):
                Start = j * 43
                End = (j + 2) * 43
                aa = asd[Start : End]
                #print(aa.shape)
                pdata.append(aa)

            #time.sleep(30)

        self.data = np.array(pdata)

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
        self.model = Sequential()

        self.model.add(Reshape((86, 128, 1), input_shape=(86, 128)))
        self.model.add(Conv2D(16, (3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(16, (3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size = (2, 2)))
        self.model.add(Dropout(0.1))

        self.model.add(Conv2D(16, (3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(16, (3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size = (2, 2)))
        self.model.add(Dropout(0.2))

        self.model.add(Conv2D(16, (3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(16, (3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size = (10, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(15, activation='softmax'))
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=["accuracy"])


    def work(self):
        self.prepare()
        self.create_model()
        self.learn()

a = Learner()
a.work()


