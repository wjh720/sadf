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
            ww = np.std(norm_asd, axis = 0)
            num_0 = np.where(ww == 0)
            qe = ww[num_0]
            #print(qe.shape)
            if (qe.shape[0] > 0):
                #print(ww)
                print(num_0)
                #print(norm_asd[:, num_0])
                time.sleep(3)
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

        f = file('data_mfcc.npy', 'r')
        self.data = np.load(f)
        f.close()

        self.label = self.label.repeat(num_repeat)
        self.label = np.eye(num_classes)[self.label]#.reshape(-1, 1, 15).repeat(num_asd, axis = 1)
        print(self.label[0, 0])

        n = self.data.shape[0]

        pdata = []
        for i in range(n):
            asd = self.data[i]

            for j in range(num_repeat):
                Start = j * 86
                End = (j + 1) * 86
                aa = asd[Start : End]
                #print(aa.shape)
                pdata.append(aa)

            #time.sleep(30)

        self.mfcc = np.array(pdata)

        print(self.data.shape)
        print(self.mfcc.shape)
        print(self.label.shape)

    def learn(self):
        tbCallBack = keras.callbacks.TensorBoard(log_dir='../Graph', histogram_freq=0, write_graph=True, write_images=True)
        checkpointer = ModelCheckpoint(filepath='/data/tmpsrt1/log_new/wph', save_best_only=True, period = 10, \
                        verbose = 1,save_weights_only = True)
        
        print(' Begin fitting ')

        self.model.fit(
            {
                'mfcc' : self.mfcc
            },
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

        def lam(X):
            print("xxxxxxxxxxxxxxxxxxxxxxxxxx")
            print(X.shape)
            X=K.max(X,axis=1)
            print(X.shape)
            return X

        self.model = Sequential()

        self.model.add(Reshape((86, 128, 1), input_shape=(86, 128)))
        self.model.add(Conv2D(64, (3, 3), padding='same',activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(64, (3, 3), padding='same',activation='relu'))
        self.model.add(MaxPooling2D(pool_size = (3, 3)))
        self.model.add(Dropout(0.1))

        self.model.add(Conv2D(128, (3, 3), padding='same',activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(128, (3, 3), padding='same',activation='relu'))
        self.model.add(MaxPooling2D(pool_size = (3, 3)))
        self.model.add(Dropout(0.15))

        self.model.add(Conv2D(128, (3, 3), padding='same',activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(128, (3, 3), padding='same',activation='relu'))

        self.model.add(Lambda(lam,output_shape=(14,128)))

        self.model.add(Dropout(0.2))
        self.model.add(Flatten())

        self.model.add(Dense(15, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])


    def create_mfcc(self):

        def lam(X):
            print("xxxxxxxxxxxxxxxxxxxxxxxxxx")
            print(X.shape)
            X=K.max(X,axis=1)
            print(X.shape)
            return X

        mfcc = Input(shape = (86, 64, ), dtype = 'float32', name = 'mfcc')

        mfcc_reshape = Reshape((86, 64, 1))(mfcc)
        Conv_1 = Conv2D(64, (3, 3), padding='same', activation='relu')
        Conv_2 = Conv2D(64, (3, 3), padding='same', activation='relu')

        conv_1 = Conv_1(mfcc_reshape)
        conv_1_bh = BatchNormalization()(conv_1)
        conv_2 = Conv_2(conv_1_bh)
        maxpool_1 = MaxPooling2D(pool_size = (3, 3))(conv_2)
        drop_1 = Dropout(0.1)(maxpool_1)

        Conv_3 = Conv2D(128, (3, 3), padding='same', activation='relu')
        Conv_4 = Conv2D(128, (3, 3), padding='same', activation='relu')

        conv_3 = Conv_3(drop_1)
        conv_3_bh = BatchNormalization()(conv_3)
        conv_4 = Conv_4(conv_3_bh)
        maxpool_2 = MaxPooling2D(pool_size = (3, 3))(conv_4)
        drop_2 = Dropout(0.15)(maxpool_2)

        Conv_5 = Conv2D(128, (3, 3), padding='same', activation='relu')
        Conv_6 = Conv2D(128, (3, 3), padding='same', activation='relu')

        conv_5 = Conv_5(drop_2)
        conv_5_bh = BatchNormalization()(conv_5)
        conv_6 = Conv_6(conv_5_bh)
        maxpool_3 = MaxPooling2D(pool_size = (3, 3))(conv_6)

        lam_1 = Lambda(lam,output_shape=(7,128))(maxpool_3)
        drop_3 = Dropout(0.2)(lam_1)

        fla_1 = Flatten()(drop_3)

        Dense_1 = Dense(15, activation='softmax', name = 'out_1')
        out = Dense_1(fla_1)

        self.model = Model(inputs = [mfcc ], outputs = [out])
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

    def predict(self):
        output = self.model.predict(       
                x = self.data,
                batch_size = 256,
                verbose = 2
            )
        print("!!!!!!!!!!!!!!!!!!!!")
        print(self.label[:10])
        print(output[:10])

    def work(self):
        self.prepare()
        #self.create_model()
        self.create_mfcc()
        self.learn()
        self.predict()

a = Learner()
a.work()


