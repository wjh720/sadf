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

num_repeat = 13
num_asd = 25
num_classes = 15
num_time = 64

class Learner():
    def __init__(self):
        pass

    def prepare(self):
        f = file('label.npy', 'r')
        self.label = np.load(f)
        f.close()

        self.label = self.label.repeat(num_repeat)
        self.label = np.eye(num_classes)[self.label]#.reshape(-1, 1, 15).repeat(num_asd, axis = 1)
        print(self.label[0, 0])

        # -----------------------------

        f_mfcc = file('data_mfcc.npy', 'r')
        self.data_mfcc = np.load(f_mfcc)
        f_mfcc.close()

        f_cqt = file('data_cqt.npy', 'r')
        self.data_cqt = np.load(f_cqt)
        f_cqt.close()

        f_ttz = file('data_ttz.npy', 'r')
        self.data_ttz = np.load(f_ttz)
        f_ttz.close()

        f_rhy = file('data_rhythm.npy', 'r')
        self.data_rhy = np.load(f_rhy)
        f_rhy.close()

        # -----------------------------

        n = self.data_mfcc.shape[0]

        mfcc = []
        cqt = []
        ttz = []
        rhy = []
        for i in range(n):
            asd_mfcc = self.data_mfcc[i]
            asd_cqt = self.data_cqt[i]
            asd_ttz = self.data_ttz[i]
            asd_rhy = self.data_rhy[i]

            for j in range(num_repeat):
                aa_mfcc = asd_mfcc[j * num_time : (j + 1) * num_time]
                aa_cqt = asd_cqt[j * num_time : (j + 1) * num_time]
                aa_ttz = asd_ttz[j * num_time : (j + 1) * num_time]
                aa_rhy = asd_rhy[j * num_time : (j + 1) * num_time]

                mfcc.append(aa_mfcc)
                cqt.append(aa_cqt)
                ttz.append(aa_ttz)
                rhy.append(aa_rhy)

        self.data_mfcc = np.array(mfcc)
        self.data_cqt = np.array(cqt)
        self.data_ttz = np.array(ttz)
        self.data_rhy = np.array(rhy)

        print(self.data_mfcc.shape)
        print(self.data_cqt.shape)
        print(self.data_ttz.shape)
        print(self.data_rhy.shape)
        print(self.label.shape)

    def learn(self):
        tbCallBack = keras.callbacks.TensorBoard(log_dir='../Graph', histogram_freq=0, write_graph=True, write_images=True)
        checkpointer = ModelCheckpoint(filepath='/data/tmpsrt1/log_new/wph', save_best_only=True, period = 10, \
                        verbose = 1,save_weights_only = True)
        
        print(' Begin fitting ')

        self.model.fit(
            x = self.data_rhy,
            y = self.label,
            batch_size = 128,
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

        self.model.add(Reshape((64, 128, 1), input_shape=(64, 128)))
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

        self.model.add(Lambda(lam,output_shape=(14, 128)))

        self.model.add(Dropout(0.2))
        self.model.add(Flatten())

        self.model.add(Dense(128, activation='softmax'))
        self.model.add(Dense(15, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])


    def create_mfcc(self):

        def lam(X):
            print("xxxxxxxxxxxxxxxxxxxxxxxxxx")
            print(X.shape)
            X=K.max(X,axis=1)
            print(X.shape)
            return X

        mfcc = Input(shape = (86, 128, ), dtype = 'float32', name = 'mel')

        mfcc_reshape = Reshape((86, 128, 1))(mfcc)
        Conv_1 = Conv2D(64, (3, 3), padding='same', activation='relu')
        #Conv_2 = Conv2D(64, (3, 3), padding='same', activation='relu')

        conv_1 = Conv_1(mfcc_reshape)
        #conv_1_bh = BatchNormalization()(conv_1)
        #conv_2 = Conv_2(conv_1_bh)
        maxpool_1 = MaxPooling2D(pool_size = (3, 3))(conv_1)
        drop_1 = Dropout(0.1)(maxpool_1)

        Conv_3 = Conv2D(64, (3, 3), padding='same', activation='relu')
        Conv_4 = Conv2D(64, (3, 3), padding='same', activation='relu')
        Conv_01 = Conv2D(64, (3, 3), padding='same', activation='relu')

        conv_3 = Conv_3(drop_1)
        conv_3_bh = BatchNormalization()(conv_3)

        concat_1 = Concatenate(axis = 3)([drop_1, conv_3_bh])
        conv_4 = Conv_4(concat_1)
        conv_4_bh = BatchNormalization()(conv_4)

        concat_2 = Concatenate(axis = 3)([drop_1, conv_3_bh, conv_4_bh])
        conv_01 = Conv_01(concat_2)

        maxpool_2 = MaxPooling2D(pool_size = (3, 3))(conv_01)
        drop_2 = Dropout(0.15)(maxpool_2)

        Conv_5 = Conv2D(64, (3, 3), padding='same', activation='relu')
        Conv_6 = Conv2D(64, (3, 3), padding='same', activation='relu')
        Conv_02 = Conv2D(64, (3, 3), padding='same', activation='relu')

        conv_5 = Conv_5(drop_2)
        conv_5_bh = BatchNormalization()(conv_5)

        concat_3 = Concatenate(axis = 3)([drop_2, conv_5_bh])
        conv_6 = Conv_6(concat_3)
        conv_6_bh = BatchNormalization()(conv_6)

        concat_4 = Concatenate(axis = 3)([drop_2, conv_5_bh, conv_6_bh])
        conv_02 = Conv_02(concat_4)

        lam_1 = Lambda(lam, output_shape=(14, 64))(conv_02)
        drop_3 = Dropout(0.2)(lam_1)

        fla_1 = Flatten()(drop_3)

        Dense_2 = Dense(128, activation = 'relu')
        Dense_1 = Dense(15, activation = 'softmax', name = 'out_1')
        den_2 = Dense_2(fla_1)
        out = Dense_1(den_2)

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
        self.create_model()
        #self.create_mfcc()
        self.learn()
        self.predict()

a = Learner()
a.work()


