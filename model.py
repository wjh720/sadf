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
si_1 = 40
si_2 = 80
si_3 = 160

class Learner():
    def __init__(self):
        pass

    def prepare(self):
        f = file('label.npy', 'r')
        self.label = np.load(f)
        f.close()

        f_1 = file('data_mfcc_1024.npy', 'r')
        f_2 = file('data_mfcc_2048.npy', 'r')
        f_3 = file('data_mfcc_4096.npy', 'r')
        self.data_1 = np.load(f_1)
        self.data_2 = np.load(f_2)
        self.data_3 = np.load(f_3)
        f.close()

        self.label = self.label.repeat(num_repeat)
        self.label = np.eye(num_classes)[self.label]#.reshape(-1, 1, 15).repeat(num_asd, axis = 1)
        print(self.label[0, 0])

        n = self.data_1.shape[0]

        pdata_1 = []
        pdata_2 = []
        pdata_3 = []
        for i in range(n):
            asd_1 = self.data_1[i]
            asd_2 = self.data_2[i]
            asd_3 = self.data_3[i]

            for j in range(num_repeat):
                aa_1 = asd_1[j * si_3 : (j + 1) * si_3]
                aa_2 = asd_2[j * si_2 : (j + 1) * si_2]
                aa_3 = asd_3[j * si_1 : (j + 1) * si_1]
                
                pdata_1.append(aa_1)
                pdata_2.append(aa_2)
                pdata_3.append(aa_3)

            #time.sleep(30)

        self.data_1 = np.array(pdata_1)
        self.data_2 = np.array(pdata_2)
        self.data_3 = np.array(pdata_3)

        print(self.data_1.shape)
        print(self.data_2.shape)
        print(self.data_3.shape)
        print(self.label.shape)

    def learn(self):
        tbCallBack = keras.callbacks.TensorBoard(log_dir='../Graph', histogram_freq=0, write_graph=True, write_images=True)
        checkpointer = ModelCheckpoint(filepath='/data/tmpsrt1/log_new/wph', save_best_only=True, period = 10, \
                        verbose = 1,save_weights_only = True)
        
        print(' Begin fitting ')

        self.model.fit(
            {
                'data_1' : self.data_1,
                'data_2' : self.data_2,
                'data_3' : self.data_3
            },
            y = self.label,
            batch_size = 64,
            epochs = 10000,
            validation_split = 0.2,
            verbose = 2,
            shuffle = True,
            callbacks = [tbCallBack,checkpointer]
        )

        print(' End fitting ')

    def create_mfcc(self):

        K_n = 5
        K_1 = 2
        K_2 = 2
        K_3 = 2

        def lam(X):
            print("xxxxxxxxxxxxxxxxxxxxxxxxxx")
            print(X.shape)
            X=K.max(X,axis=1)
            print(X.shape)
            return X

        mfcc_1 = Input(shape = (si_1, 64, ), dtype = 'float32', name = 'data_3')
        mfcc_2 = Input(shape = (si_2, 64, ), dtype = 'float32', name = 'data_2')
        mfcc_3 = Input(shape = (si_3, 64, ), dtype = 'float32', name = 'data_1')

        mfcc_1_r = Reshape((si_1, 64, 1))(mfcc_1)
        mfcc_2_r = Reshape((si_2, 64, 1))(mfcc_2)
        mfcc_3_r = Reshape((si_3, 64, 1))(mfcc_3)


        Conv_1_1 = Conv2D(64, (K_n, K_n), padding='same', activation='relu')
        Conv_1_2 = Conv2D(64, (K_n, K_n), padding='same', activation='relu')
        Conv_2_1 = Conv2D(64, (K_n, K_n), padding='same', activation='relu')
        Conv_2_2 = Conv2D(64, (K_n, K_n), padding='same', activation='relu')
        Conv_3_1 = Conv2D(64, (K_n, K_n), padding='same', activation='relu')
        Conv_3_2 = Conv2D(64, (K_n, K_n), padding='same', activation='relu')

        conv_1_1 = Conv_1_1(mfcc_1_r)
        conv_1_2 = Conv_1_2(conv_1_1)
        conv_2_1 = Conv_2_1(mfcc_2_r)
        conv_2_2 = Conv_2_2(conv_2_1)
        conv_3_1 = Conv_3_1(mfcc_3_r)
        conv_3_2 = Conv_3_2(conv_3_1)
        conv_1_d = BatchNormalization()(conv_1_2)
        conv_2_d = BatchNormalization()(conv_2_2)
        conv_3_d = BatchNormalization()(conv_3_2)

        #-----------------------------------

        in1_conv_1_1 = MaxPooling2D(pool_size = (K_1, K_1))(conv_1_d)
        in1_conv_1_2 = MaxPooling2D(pool_size = (1, K_1))(conv_1_d)

        in1_conv_2_2 = MaxPooling2D(pool_size = (K_1, K_1))(conv_2_d)
        in1_conv_2_1 = MaxPooling2D(pool_size = (K_1, 1))(in1_conv_2_2)
        in1_conv_2_3 = MaxPooling2D(pool_size = (1, K_1))(conv_2_d)

        in1_conv_3_3 = MaxPooling2D(pool_size = (K_1, K_1))(conv_3_d)
        in1_conv_3_2 = MaxPooling2D(pool_size = (K_1, 1))(in1_conv_3_3)

        conv_1_in_1 = Concatenate(axis = 3)([in1_conv_1_1, in1_conv_2_1])
        conv_2_in_1 = Concatenate(axis = 3)([in1_conv_2_2, in1_conv_1_2, in1_conv_3_2])
        conv_3_in_1 = Concatenate(axis = 3)([in1_conv_3_3, in1_conv_2_3])

        #-----------------------------------

        Conv_1_3 = Conv2D(128, (K_n, K_n), padding='same', activation='relu')
        Conv_1_4 = Conv2D(128, (K_n, K_n), padding='same', activation='relu')
        Conv_2_3 = Conv2D(128, (K_n, K_n), padding='same', activation='relu')
        Conv_2_4 = Conv2D(128, (K_n, K_n), padding='same', activation='relu')
        Conv_3_3 = Conv2D(128, (K_n, K_n), padding='same', activation='relu')
        Conv_3_4 = Conv2D(128, (K_n, K_n), padding='same', activation='relu')

        conv_1_3 = Conv_1_3(conv_1_in_1)
        conv_1_4 = Conv_1_4(conv_1_3)
        conv_2_3 = Conv_2_3(conv_2_in_1)
        conv_2_4 = Conv_2_4(conv_2_3)
        conv_3_3 = Conv_3_3(conv_3_in_1)
        conv_3_4 = Conv_3_4(conv_3_3)
        conv_1_dd = BatchNormalization()(conv_1_4)
        conv_2_dd = BatchNormalization()(conv_2_4)
        conv_3_dd = BatchNormalization()(conv_3_4)

        #-----------------------------------

        in2_conv_1_1 = MaxPooling2D(pool_size = (K_2, K_2))(conv_1_dd)
        in2_conv_1_2 = MaxPooling2D(pool_size = (1, K_2))(conv_1_dd)

        in2_conv_2_2 = MaxPooling2D(pool_size = (K_2, K_2))(conv_2_dd)
        in2_conv_2_1 = MaxPooling2D(pool_size = (K_2, 1))(in2_conv_2_2)
        in2_conv_2_3 = MaxPooling2D(pool_size = (1, K_2))(conv_2_dd)

        in2_conv_3_3 = MaxPooling2D(pool_size = (K_2, K_2))(conv_3_dd)
        in2_conv_3_2 = MaxPooling2D(pool_size = (K_2, 1))(in2_conv_3_3)

        conv_1_in_2 = Concatenate(axis = 3)([in2_conv_1_1, in2_conv_2_1])
        conv_2_in_2 = Concatenate(axis = 3)([in2_conv_2_2, in2_conv_1_2, in2_conv_3_2])
        conv_3_in_2 = Concatenate(axis = 3)([in2_conv_3_3, in2_conv_2_3])

        #-----------------------------------

        Conv_1_5 = Conv2D(128, (K_n, K_n), padding='same', activation='relu')
        Conv_1_6 = Conv2D(128, (K_n, K_n), padding='same', activation='relu')
        Conv_2_5 = Conv2D(128, (K_n, K_n), padding='same', activation='relu')
        Conv_2_6 = Conv2D(128, (K_n, K_n), padding='same', activation='relu')
        Conv_3_5 = Conv2D(128, (K_n, K_n), padding='same', activation='relu')
        Conv_3_6 = Conv2D(128, (K_n, K_n), padding='same', activation='relu')

        conv_1_5 = Conv_1_5(conv_1_in_2)
        conv_1_6 = Conv_1_6(conv_1_5)
        conv_2_5 = Conv_2_5(conv_2_in_2)
        conv_2_6 = Conv_2_6(conv_2_5)
        conv_3_5 = Conv_3_5(conv_3_in_2)
        conv_3_6 = Conv_3_6(conv_3_5)
        conv_1_ddd = BatchNormalization()(conv_1_6)
        conv_2_ddd = BatchNormalization()(conv_2_6)
        conv_3_ddd = BatchNormalization()(conv_3_6)

        #-----------------------------------

        in3_conv_1_2 = MaxPooling2D(pool_size = (1, K_3))(conv_1_ddd)

        in3_conv_2_2 = MaxPooling2D(pool_size = (K_3, K_3))(conv_2_ddd)

        in3_conv_3_3 = MaxPooling2D(pool_size = (K_3, K_3))(conv_3_ddd)
        in3_conv_3_2 = MaxPooling2D(pool_size = (K_3, 1))(in3_conv_3_3)

        conv_2_in_3 = Concatenate(axis = 3)([in3_conv_2_2, in3_conv_1_2, in3_conv_3_2])

        #-----------------------------------

        Conv_1 = Conv2D(128, (K_n, K_n), padding='same', activation='relu')
        Conv_2 = Conv2D(128, (K_n, K_n), padding='same', activation='relu')

        conv2_1 = Conv_1(conv_2_in_3)
        conv2_2 = Conv_2(conv2_1)

        lam_1 = Lambda(lam, output_shape=(32, 128))(conv2_2)
        drop = Dropout(0.2)(lam_1)

        fla_1 = Flatten()(drop)

        Dense_2 = Dense(128, activation = 'relu')
        Dense_1 = Dense(15, activation = 'softmax', name = 'out_1')
        den_2 = Dense_2(fla_1)
        out = Dense_1(den_2)

        self.model = Model(inputs = [mfcc_1, mfcc_2, mfcc_3], outputs = [out])
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
        self.create_mfcc()
        self.learn()
        self.predict()

a = Learner()
a.work()


