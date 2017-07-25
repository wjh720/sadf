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

num_repeat = 10
num_asd = 25
num_classes = 15

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

        self.label = self.label.repeat(num_repeat)
        self.label = np.eye(num_classes)[self.label]#.reshape(-1, 1, 15).repeat(num_asd, axis = 1)
        print(self.label[0, 0])

        n = self.data.shape[0]

        pdata = []
        for i in range(n):
            asd = self.data[i]
            #print(asd.shape)
            norm_asd = asd - np.mean(asd, axis = 0)
            #print(np.mean(norm_asd, axis = 0))
            #print(np.std(norm_asd, axis = 0))
            #norm_asd = norm_asd / np.std(norm_asd, axis = 0)
            #print(np.std(norm_asd, axis = 0))

            #time.sleep(30)

            for j in range(num_repeat):
                Start = j * 86
                End = (j + 1) * 86
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

        def my_loss(y_true, y_pred):
            print("asavad ",y_true.get_shape)
            print("asavad ",y_pred.get_shape)

            ans = []
            for i in range(num_asd):
                ans.append(K.categorical_crossentropy(y_true[:, i], y_pred[:,i]))
            return K.mean(tf.stack(ans))

        def mode(y_true, y_pred):
            a = K.argmax(y_pred, axis = 2)
            aa = K.argmax(y_true, axis = 2)
            ans = []
            for i in range(15):
                tmp = K.equal(a, i)
                b = tf.reduce_sum(tf.cast(tmp, tf.int32), axis = 1, keep_dims=False)
                ans.append(b)
            c = tf.stack(ans)
            d = K.argmax(K.transpose(c), axis = 1)
            return K.mean(K.equal(aa[:, 0], d))

        def mean_acc(y_true, y_pred):
            a = K.argmax(y_pred, axis = 2)
            aa = K.argmax(y_true, axis = 2)

            tmp = K.equal(a, aa)
            return K.mean(K.mean(tf.cast(tmp, tf.float32)))

        def lam(X):
            print("xxxxxxxxxxxxxxxxxxxxxxxxxx")
            print(X.shape)
            X=K.max(X,axis=1)
            print(X.shape)
            return X

        self.model = Sequential()

        self.model.add(Reshape((86, 128, 1), input_shape=(86, 128)))
        self.model.add(Conv2D(64, (5, 5), padding='same',activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(64, (5, 5), padding='same',activation='relu'))
        #self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size = (5, 5)))
        self.model.add(Dropout(0.1))

        self.model.add(Conv2D(128, (5, 5), padding='same',activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(128, (5, 5), padding='same',activation='relu'))
        #self.model.add(BatchNormalization())
        self.model.add(Lambda(lam,output_shape=(25,128)))
        #self.model.add(MaxPooling2D(pool_size = (10, 2)))
        self.model.add(Dropout(0.2))
        self.model.add(Flatten())
        #self.model.add(Conv1D(256, 1, padding='same', activation='relu'))
        #self.model.add(Conv1D(15, 1, padding='same', activation='softmax'))
        #self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(15, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])
        #self.model.compile(loss = my_loss, optimizer='adam',metrics=[mean_acc, mode])


    def work(self):
        self.prepare()
        self.create_model()
        #self.log_model_summary()
        self.learn()

a = Learner()
a.work()


