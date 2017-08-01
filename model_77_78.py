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
from keras.layers import Conv1D, MaxPooling1D,Conv2D,MaxPooling2D, AveragePooling2D, UpSampling2D
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
si_1 = 16
si_2 = 64
si_3 = 256
num_time = 64

class Learner():
    def __init__(self):
        pass

    def prepare_3(self):
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

        # -----------------------------

        n = self.data_mfcc.shape[0]

        mfcc = []
        cqt = []
        ttz = []
        for i in range(n):
            asd_mfcc = self.data_mfcc[i]
            asd_cqt = self.data_cqt[i]
            asd_ttz = self.data_ttz[i]

            for j in range(num_repeat):
                aa_mfcc = asd_mfcc[j * num_time : (j + 1) * num_time]
                aa_cqt = asd_cqt[j * num_time : (j + 1) * num_time]
                aa_ttz = asd_ttz[j * num_time : (j + 1) * num_time]
                mfcc.append(aa_mfcc)
                cqt.append(aa_cqt)
                ttz.append(aa_ttz)

        self.data_mfcc = np.array(mfcc)
        self.data_cqt = np.array(cqt)
        self.data_ttz = np.array(ttz)

        print(self.data_mfcc.shape)
        print(self.data_cqt.shape)
        print(self.data_ttz.shape)
        print(self.label.shape)

    def prepare(self):
        f = file('label.npy', 'r')
        self.label = np.load(f)
        f.close()

        f_1 = file('data_mfcc_1024.npy', 'r')
        f_2 = file('data_mfcc_2048.npy', 'r')
        f_3 = file('data_mfcc_4096.npy', 'r')
        ff = file('data.npy', 'r')
        self.data_1 = np.load(f_1)
        self.data_2 = np.load(f_2)
        self.data_3 = np.load(f_3)
        self.data_mel = np.load(ff)
        f.close()

        self.label = self.label.repeat(num_repeat)
        self.label = np.eye(num_classes)[self.label]#.reshape(-1, 1, 15).repeat(num_asd, axis = 1)
        print(self.label[0, 0])

        print('-----------------')
        print(self.data_mel.shape)
        print('-----------------')

        n = self.data_1.shape[0]

        pdata_1 = []
        pdata_2 = []
        pdata_3 = []
        mdata = []
        for i in range(n):
            asd_1 = self.data_1[i]
            asd_2 = self.data_2[i]
            asd_3 = self.data_3[i]
            asd_m = self.data_mel[i]

            for j in range(num_repeat):
                aa_1 = asd_1[j * si_3 : (j + 1) * si_3]
                aa_2 = asd_2[j * si_2 : (j + 1) * si_2]
                aa_3 = asd_3[j * si_1 : (j + 1) * si_1]
                aa_m = asd_m[j * si_2 : (j + 1) * si_2]
                
                pdata_1.append(aa_1)
                pdata_2.append(aa_2)
                pdata_3.append(aa_3)
                mdata.append(aa_m)

            #time.sleep(30)

        self.data_1 = np.array(pdata_1)
        self.data_2 = np.array(pdata_2)
        self.data_3 = np.array(pdata_3)
        self.data_mel = np.array(mdata)

        print(self.data_1.shape)
        print(self.data_2.shape)
        print(self.data_3.shape)
        print(self.data_mel.shape)
        print(self.label.shape)
        print('----------------')

    def learn(self):
        tbCallBack = keras.callbacks.TensorBoard(log_dir='../Graph_7778', histogram_freq=0, write_graph=True, write_images=True)
        checkpointer = ModelCheckpoint(filepath='/data/tmpsrt1/log_new/weights.{epoch:02d}-{val_loss:.2f}.hdf5', \
                        period = 1, verbose = 1,save_weights_only = True)
        
        print(' Begin fitting ')

        self.model.fit(
            {
                'data_4096' : self.data_3,  # 16, 64
                'data_cqt' : self.data_cqt, # 64, 64
                #'data_mel' : self.data_mel  # 64, 128
                'data_2048' : self.data_2
            },
            {
                'out_1' : self.label,
                'out_2' : self.label,
                'out_3' : self.label
            },
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
        K_11 = 4
        K_1 = 2
        K_2 = 2
        K_3 = 2

        def lam(X):
            print("xxxxxxxxxxxxxxxxxxxxxxxxxx")
            print(X.shape)
            X=K.max(X,axis=1)
            print(X.shape)
            return X

        mfcc_1 = Input(shape = (si_1, 64, ), dtype = 'float32', name = 'data_4096')
        mfcc_2 = Input(shape = (si_2, 64, ), dtype = 'float32', name = 'data_cqt')
        mfcc_3 = Input(shape = (si_2, 64, ), dtype = 'float32', name = 'data_2048')

        mfcc_1_r = Reshape((si_1, 64, 1))(mfcc_1)
        mfcc_2_r = Reshape((si_2, 64, 1))(mfcc_2)
        mfcc_3_r = Reshape((si_2, 64, 1))(mfcc_3)

        # -----------------------------
        '''
        Conv_03_1 = Conv2D(64, (K_n, K_n), padding='same', activation='relu')
        Conv_03_2 = Conv2D(64, (K_n, K_n), padding='same', activation='relu')

        conv_03_1 = Conv_03_1(mfcc_3_r)
        conv_03_2 = Conv_03_2(conv_03_1)
        conv_03_d = BatchNormalization()(conv_03_2)
        conv_03 = MaxPooling2D(pool_size = (1, 2))(conv_03_d)
        conv_03_ok = Dropout(0.05)(conv_03)
        '''
        # -----------------------------


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
        in1_conv_1_2 = UpSampling2D(size = (4, 1))(in1_conv_1_1)
        in1_conv_1_3 = in1_conv_1_2

        in1_conv_2_2 = MaxPooling2D(pool_size = (K_1, K_1))(conv_2_d)
        in1_conv_2_1 = MaxPooling2D(pool_size = (K_11, 1))(in1_conv_2_2)
        in1_conv_2_3 = in1_conv_2_2

        in1_conv_3_3 = MaxPooling2D(pool_size = (K_1, K_1))(conv_3_d)
        in1_conv_3_2 = in1_conv_3_3
        in1_conv_3_1 = MaxPooling2D(pool_size = (K_11, 1))(in1_conv_3_3)

        conv_1_in_1_d = Concatenate(axis = 3)([in1_conv_1_1, in1_conv_2_1, in1_conv_3_1])
        conv_2_in_1_d = Concatenate(axis = 3)([in1_conv_2_2, in1_conv_1_2, in1_conv_3_2])
        conv_3_in_1_d = Concatenate(axis = 3)([in1_conv_3_3, in1_conv_1_3, in1_conv_2_3])

        conv_1_in_1 = Dropout(0.1)(conv_1_in_1_d)
        conv_2_in_1 = Dropout(0.1)(conv_2_in_1_d)
        conv_3_in_1 = Dropout(0.1)(conv_3_in_1_d)

        #-----------------------------------

        Conv_1_3 = Conv2D(64, (K_n, K_n), padding='same', activation='relu')
        Conv_1_4 = Conv2D(64, (K_n, K_n), padding='same', activation='relu')
        Conv_2_3 = Conv2D(64, (K_n, K_n), padding='same', activation='relu')
        Conv_2_4 = Conv2D(64, (K_n, K_n), padding='same', activation='relu')
        Conv_3_3 = Conv2D(64, (K_n, K_n), padding='same', activation='relu')
        Conv_3_4 = Conv2D(64, (K_n, K_n), padding='same', activation='relu')

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
        in2_conv_1_2 = UpSampling2D(size = (4, 1))(in2_conv_1_1)
        in2_conv_1_3 = in2_conv_1_2

        in2_conv_2_2 = MaxPooling2D(pool_size = (K_2, K_2))(conv_2_dd)
        in2_conv_2_1 = MaxPooling2D(pool_size = (K_11, 1))(in2_conv_2_2)
        in2_conv_2_3 = in2_conv_2_2

        in2_conv_3_3 = MaxPooling2D(pool_size = (K_2, K_2))(conv_3_dd)
        in2_conv_3_2 = in2_conv_3_3
        in2_conv_3_1 = MaxPooling2D(pool_size = (K_11, 1))(in2_conv_3_3)

        conv_1_in_2_d = Concatenate(axis = 3)([in2_conv_1_1, in2_conv_2_1, in2_conv_3_1])
        conv_2_in_2_d = Concatenate(axis = 3)([in2_conv_2_2, in2_conv_1_2, in2_conv_3_2])
        conv_3_in_2_d = Concatenate(axis = 3)([in2_conv_3_3, in2_conv_1_3, in2_conv_2_3])

        conv_1_in_2 = Dropout(0.15)(conv_1_in_2_d)
        conv_2_in_2 = Dropout(0.15)(conv_2_in_2_d)
        conv_3_in_2 = Dropout(0.15)(conv_3_in_2_d)

        #-----------------------------------

        Conv_1_5 = Conv2D(64, (K_n, K_n), padding='same', activation='relu')
        Conv_1_6 = Conv2D(64, (K_n, K_n), padding='same', activation='relu')
        Conv_2_5 = Conv2D(64, (K_n, K_n), padding='same', activation='relu')
        Conv_2_6 = Conv2D(64, (K_n, K_n), padding='same', activation='relu')
        Conv_3_5 = Conv2D(64, (K_n, K_n), padding='same', activation='relu')
        Conv_3_6 = Conv2D(64, (K_n, K_n), padding='same', activation='relu')

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

        in3_conv_1_1 = MaxPooling2D(pool_size = (K_3, K_3))(conv_1_ddd)
        in3_conv_1_2 = UpSampling2D(size = (4, 1))(in3_conv_1_1)
        in3_conv_1_3 = in3_conv_1_2

        in3_conv_2_2 = MaxPooling2D(pool_size = (K_3, K_3))(conv_2_ddd)
        in3_conv_2_1 = MaxPooling2D(pool_size = (K_11, 1))(in3_conv_2_2)
        in3_conv_2_3 = in3_conv_2_2

        in3_conv_3_3 = MaxPooling2D(pool_size = (K_3, K_3))(conv_3_ddd)
        in3_conv_3_2 = in3_conv_3_3
        in3_conv_3_1 = MaxPooling2D(pool_size = (K_11, 1))(in3_conv_3_3)

        conv_1_in_3_d = Concatenate(axis = 3)([in3_conv_1_1, in3_conv_2_1, in3_conv_3_1])
        conv_2_in_3_d = Concatenate(axis = 3)([in3_conv_2_2, in3_conv_1_2, in3_conv_3_2])
        conv_3_in_3_d = Concatenate(axis = 3)([in3_conv_3_3, in3_conv_1_3, in3_conv_2_3])

        conv_1_in_3 = Dropout(0.2)(conv_1_in_3_d)
        conv_2_in_3 = Dropout(0.2)(conv_2_in_3_d)
        conv_3_in_3 = Dropout(0.2)(conv_3_in_3_d)

        #-----------------------------------

        Conv_1_7 = Conv2D(64, (K_n, K_n), padding='same', activation='relu')
        Conv_1_8 = Conv2D(64, (K_n, K_n), padding='same', activation='relu')
        Conv_2_7 = Conv2D(64, (K_n, K_n), padding='same', activation='relu')
        Conv_2_8 = Conv2D(64, (K_n, K_n), padding='same', activation='relu')
        Conv_3_7 = Conv2D(64, (K_n, K_n), padding='same', activation='relu')
        Conv_3_8 = Conv2D(64, (K_n, K_n), padding='same', activation='relu')

        conv_1_7 = Conv_1_7(conv_1_in_3)
        conv_1_8 = Conv_1_8(conv_1_7)
        conv_2_7 = Conv_2_7(conv_2_in_3)
        conv_2_8 = Conv_2_8(conv_2_7)
        conv_3_7 = Conv_3_7(conv_3_in_3)
        conv_3_8 = Conv_3_8(conv_3_7)

        lam_1 = Lambda(lam, output_shape=(8, 64))(conv_1_8)
        lam_2 = Lambda(lam, output_shape=(8, 64))(conv_2_8)
        lam_3 = Lambda(lam, output_shape=(8, 64))(conv_3_8)
        drop_1 = Dropout(0.3)(lam_1)
        drop_2 = Dropout(0.3)(lam_2)
        drop_3 = Dropout(0.3)(lam_3)

        fla_1 = Flatten()(drop_1)
        fla_2 = Flatten()(drop_2)
        fla_3 = Flatten()(drop_3)

        Dense_1_1 = Dense(128, activation = 'relu')
        Dense_2_1 = Dense(128, activation = 'relu')
        Dense_3_1 = Dense(128, activation = 'relu')
        Dense_1_2 = Dense(15, activation = 'softmax', name = 'out_1')
        Dense_2_2 = Dense(15, activation = 'softmax', name = 'out_2')
        Dense_3_2 = Dense(15, activation = 'softmax', name = 'out_3')

        den_1_1 = Dense_1_1(fla_1)
        den_1_2 = Dense_1_2(den_1_1)
        den_2_1 = Dense_2_1(fla_2)
        den_2_2 = Dense_2_2(den_2_1)
        den_3_1 = Dense_3_1(fla_3)
        den_3_2 = Dense_3_2(den_3_1)

        self.model = Model(inputs = [mfcc_1, mfcc_2, mfcc_3], outputs = [den_1_2, den_2_2, den_3_2])
        self.model.compile(loss = {'out_1' : 'categorical_crossentropy', 'out_2' : 'categorical_crossentropy', \
                            'out_3' : 'categorical_crossentropy'}, \
                            loss_weights = {'out_1' : 1., 'out_2' : 1., 'out_3' : 1.}, \
                            optimizer = 'adam', \
                            metrics = {'out_1' : 'accuracy', 'out_2' : 'accuracy', 'out_3' : 'accuracy'})

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
        self.prepare_3()
        self.create_mfcc()
        self.learn()
        self.predict()

a = Learner()
a.work()


