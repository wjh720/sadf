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
si_3 = 32

path = '../data/TUT-acoustic-scenes-2017-development/'

class Learner():
    def __init__(self):
        pass

    def prepare_label(self, data):
        #print(data[:22])
        data = data.repeat(num_repeat)
        data = np.eye(num_classes)[data]
        #print(data.shape)
        return data

    def Load_1(self, name1, name2):
        data = []

        f = file(name1 + '_train' + name2, 'r')
        data.append(np.load(f))
        f.close()

        f = file(name1 + '_evaluate' + name2, 'r')
        data.append(np.load(f))
        f.close()

        #print('----------------')
        data[0] = self.prepare_label(data[0])
        data[1] = self.prepare_label(data[1])
        #print('----------------')

        return data

    def prepare_data(self, data, length):
        n = data.shape[0]
        pdata = []

        for i in range(n):
            asd = data[i]
            for j in range(num_repeat):
                aa = asd[j * length : (j + 1) * length]
                pdata.append(aa)
                #pdata.append(aa[::-1])
        pdata = np.array(pdata)
        #print(pdata.shape)
        return pdata

    def Load_2(self, name1, name2, length):
        data = []

        f = file(name1 + '_train' + name2, 'r')
        data.append(np.load(f))
        f.close()

        f = file(name1 + '_evaluate' + name2 , 'r')
        data.append(np.load(f))
        f.close()

        #print('----------------')
        data[0] = self.prepare_data(data[0], length)
        data[1] = self.prepare_data(data[1], length)
        #print('----------------')

        return data

    def prepare(self, fol): 
        meta_path = path + 'evaluation_setup/'
        name = meta_path + ('fold%d' % fol)

        self.label = self.Load_1(name, '_label')
        self.data_cqt = self.Load_2(name, '_data_cqt', si_2)
        self.data_2048 = self.Load_2(name, '_data_2048', si_2)
        self.data_4096 = self.Load_2(name, '_data_4096', si_3)
        self.data_8192 = self.Load_2(name, '_data_8192', si_1)
        self.data_mel = self.Load_2(name, '_data_mel', si_2)

        if (fol == 0):
            print('----------------')
            print(self.label[0].shape)
            print(self.data_cqt[0].shape)
            print(self.data_2048[0].shape)
            print(self.data_4096[0].shape)
            print(self.data_8192[0].shape)
            print(self.data_mel[0].shape)
            print('----------------')


    def learn(self, fol):
        tbCallBack = keras.callbacks.TensorBoard(log_dir='../Graph_merge2_fold%d' % fol, \
                            histogram_freq=0, write_graph=True, write_images=True)
        checkpointer = ModelCheckpoint(filepath='/data/tmpsrt1/log_new/weights_merge2_fold%d.{epoch:02d}.hdf5' % fol, \
                        period = 1, verbose = 1, save_weights_only = True)
        
        print(' Begin fitting %d' % fol)

        self.x_data = {
            'data_8192' : self.data_8192[0],
            'data_cqt' : self.data_cqt[0],
            'data_2048' : self.data_2048[0]
        }
        self.y_data = {
            'out_1' : self.label[0],
            'out_2' : self.label[0],
            'out_3' : self.label[0]
        }

        self.valid_data = (
            {
                'data_8192' : self.data_8192[1],
                'data_cqt' : self.data_cqt[1],
                'data_2048' : self.data_2048[1]
            }, \
            {
                'out_1' : self.label[1],
                'out_2' : self.label[1],
                'out_3' : self.label[1]
            }
        )

        self.model.fit(
            x = self.x_data,
            y = self.y_data,
            validation_data = self.valid_data,
            batch_size = 64,
            epochs = 40,
            verbose = 2,
            shuffle = True,
            callbacks = [tbCallBack,checkpointer]
        )

        print(' End fitting %d' % fol)

    def create_mfcc(self):

        K_n = 5
        K_11 = 4
        K_1 = 2
        K_2 = 2
        K_3 = 2
        size = 64

        def lam(X):
            print("xxxxxxxxxxxxxxxxxxxxxxxxxx")
            print(X.shape)
            X=K.max(X,axis=1)
            print(X.shape)
            return X

        mfcc_1 = Input(shape = (si_1, 64, ), dtype = 'float32', name = 'data_8192')
        mfcc_2 = Input(shape = (si_2, 64, ), dtype = 'float32', name = 'data_cqt')
        mfcc_3 = Input(shape = (si_2, 64, ), dtype = 'float32', name = 'data_2048')

        mfcc_1_r = Reshape((si_1, 64, 1))(mfcc_1)
        mfcc_2_r = Reshape((si_2, 64, 1))(mfcc_2)
        mfcc_3_r = Reshape((si_2, 64, 1))(mfcc_3)

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
        '''
        in1_conv_1_1 = MaxPooling2D(pool_size = (K_1, K_1))(conv_1_d)
        in1_conv_2_2 = MaxPooling2D(pool_size = (K_1, K_1))(conv_2_d)
        in1_conv_3_3 = MaxPooling2D(pool_size = (K_1, K_1))(conv_3_d)
        '''
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

        Conv_1_5 = Conv2D(size, (K_n, K_n), padding='same', activation='relu')
        Conv_1_6 = Conv2D(size, (K_n, K_n), padding='same', activation='relu')
        Conv_2_5 = Conv2D(size, (K_n, K_n), padding='same', activation='relu')
        Conv_2_6 = Conv2D(size, (K_n, K_n), padding='same', activation='relu')
        Conv_3_5 = Conv2D(size, (K_n, K_n), padding='same', activation='relu')
        Conv_3_6 = Conv2D(size, (K_n, K_n), padding='same', activation='relu')

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

        Conv_1_7 = Conv2D(size, (K_n, K_n), padding='same', activation='relu')
        Conv_1_8 = Conv2D(size, (K_n, K_n), padding='same', activation='relu')
        Conv_2_7 = Conv2D(size, (K_n, K_n), padding='same', activation='relu')
        Conv_2_8 = Conv2D(size, (K_n, K_n), padding='same', activation='relu')
        Conv_3_7 = Conv2D(size, (K_n, K_n), padding='same', activation='relu')
        Conv_3_8 = Conv2D(size, (K_n, K_n), padding='same', activation='relu')

        conv_1_7 = Conv_1_7(conv_1_in_1)
        conv_1_8 = Conv_1_8(conv_1_7)
        conv_2_7 = Conv_2_7(conv_2_in_1)
        conv_2_8 = Conv_2_8(conv_2_7)
        conv_3_7 = Conv_3_7(conv_3_in_1)
        conv_3_8 = Conv_3_8(conv_3_7)

        lam_1 = Lambda(lam, output_shape=(32, size))(conv_1_8)
        lam_2 = Lambda(lam, output_shape=(32, size))(conv_2_8)
        lam_3 = Lambda(lam, output_shape=(32, size))(conv_3_8)
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

        def Calc(label, data):
            num = np.argmax(label, axis = 1)

            if (np.max(num) != np.min(num)):
                print('calc error!')

            asd = np.argmax(data, axis = 1)

            counts = np.bincount(asd)
            ans = np.argmax(counts)

            #asd = np.sum(data, axis = 0)
            #ans = np.argmax(asd)

            '''
            if (ans != num[0]):
                print([self.dict_class[num[0]]])
                print([self.dict_class[ans]])
                print('--------------------')
                print([(self.dict_class[i], counts[i]) for i in range(counts.shape[0])])
                print('--------------------')
                print([self.dict_class[x] for x in asd[:50]])
                print('--------------------')
                #time.sleep(5)
            ''' 
            self.total_wise[num[0]] = self.total_wise[num[0]] + 1
            self.ans_wise[num[0]] = self.ans_wise[num[0]] + (ans == num[0])

            return float(ans == num[0])

        meta_path = path + 'evaluation_setup/'
        self.create_mfcc()

        acc = []
        acc_wise = {}

        for fol in range(1, 5):
            filename = '/data/tmpsrt1/log_new/weights_merge_fold%d.29.hdf5' % fol
            self.model.load_weights(filename)
            self.prepare(fol)

            self.valid_data = (
                {
                    'data_8192' : self.data_8192[1],
                    'data_cqt' : self.data_cqt[1],
                    'data_2048' : self.data_2048[1]
                }, \
                {
                    'out_1' : self.label[1],
                    'out_2' : self.label[1],
                    'out_3' : self.label[1]
                }
            )

            output = self.model.predict(       
                    x = self.valid_data[0],
                    batch_size = 64,
                    verbose = 2
                )
            print("!!!!!!!!!!!!!!!!!!!!")

            label = self.valid_data[1]['out_1']

            fold_name = meta_path + ('fold%d' % fol)
            load_name = fold_name + '_evaluate.txt'

            dict_label = {}
            self.dict_class = {}
            num_label = 0
            with open(load_name, 'r') as ff:
                for line in ff:
                    parts = line.split('\t')

                    if (parts[1] not in dict_label):
                        dict_label[parts[1]] = num_label
                        self.dict_class[num_label] = parts[1]
                        num_label = num_label + 1

            n = label.shape[0] / num_repeat
            ans = []
            self.ans_wise = np.zeros(15)
            self.total_wise = np.zeros(15)
            for i in range(n):
                asd = label[i * num_repeat : (i + 1) * num_repeat]
                data_1 = output[0][i * num_repeat : (i + 1) * num_repeat]
                data_2 = output[1][i * num_repeat : (i + 1) * num_repeat]
                data_3 = output[2][i * num_repeat : (i + 1) * num_repeat]

                data_asd = np.concatenate([data_1, data_2, data_3], axis = 0)
                res = Calc(asd, data_asd)
                ans.append(res)

            acc_fol = np.mean(np.array(ans))
            print(acc_fol)
            acc.append(acc_fol)
            print('----------------------')

            for i in range(15):
                name = self.dict_class[i]
                acc_i = 1. * self.ans_wise[i] / self.total_wise[i]
                if (fol == 1):
                    acc_wise[name] = []
                acc_wise[name].append(acc_i)

            print('----------------------')

        print('totoal_acc : %lf' % np.mean(acc))
        for item in acc_wise:
            print(item)
            print(np.mean(np.array(acc_wise[item])))

    def evaluation(self):
        def Load(name1, name2, length):
            f = file(name1 + name2, 'r')
            data = np.load(f)
            f.close()

            data = self.prepare_data(data, length)

            return data
        
        path1 = '../data/TUT-acoustic-scenes-2017-development/'
        meta_path = path1 + 'evaluation_setup/'
        self.dict = []

        for fol in range(1, 5):
            filename = '/data/tmpsrt1/log_new/weights_merge_fold%d.29.hdf5' % fol
            fold_name = meta_path + ('fold%d' % fol)
            load_name = fold_name + '_evaluate.txt'

            dict_label = {}
            self.dict_class = {}
            num_label = 0
            with open(load_name, 'r') as ff:
                for line in ff:
                    parts = line.split('\t')

                    if (parts[1] not in dict_label):
                        dict_label[parts[1]] = num_label
                        self.dict_class[num_label] = parts[1]
                        num_label = num_label + 1

            self.dict.append(self.dict_class)
            if (fol == 1):
                self.dict_label = dict_label

        path2 = '../data/TUT-acoustic-scenes-2017-evaluation/'
        meta_path = path2 + 'evaluation_setup/'
        name = meta_path + 'test.txt_test'
        
        self.data_cqt = Load(name, '_data_cqt', si_2)
        self.data_2048 = Load(name, '_data_2048', si_2)
        self.data_4096 = Load(name, '_data_4096', si_3)
        self.data_8192 = Load(name, '_data_8192', si_1)
        self.data_mel = Load(name, '_data_mel', si_2)

        self.create_mfcc()

        output = []

        for fol in range(1, 5):
            filename = '/data/tmpsrt1/log_new/weights_merge_fold%d.29.hdf5' % fol
            self.model.load_weights(filename)

            self.valid_data = {
                    'data_8192' : self.data_8192,
                    'data_cqt' : self.data_cqt,
                    'data_2048' : self.data_2048
                }
            

            output1 = self.model.predict(       
                    x = self.valid_data,
                    batch_size = 64,
                    verbose = 2
                )

            output.append(output1)

        n = output[0][0].shape[0] / num_repeat
        res = []
        for i in range(n):
            af = []
            for j in range(4):
                data_1 = output[j][0][i * num_repeat : (i + 1) * num_repeat]
                data_2 = output[j][1][i * num_repeat : (i + 1) * num_repeat]
                data_3 = output[j][2][i * num_repeat : (i + 1) * num_repeat]
                data_asd = np.concatenate([data_1, data_2, data_3], axis = 0)

                asd = np.argmax(data_asd, axis = 1)
                vas = [self.dict[j][x] for x in asd]
                #print(vas)
                #time.sleep(10)
                wqe = [self.dict_label[x] for x in vas]
                af.append(wqe)

            af = np.concatenate(af)
            #print(af)
            counts = np.bincount(af)
            ans = np.argmax(counts)
            res.append(ans)
            #print(counts)
            #print(ans)

            #time.sleep(100)
            #print('---------------------')

        save_name = 'result.txt'
        f_name = meta_path + 'test.txt'
        f_list = []
        with open(f_name, 'r') as ff:
            for line in ff:
                parts = line.split('\t')
                parts = parts[0].split('\r\n')
                f_list.append(parts[0])

        with open(save_name, 'w') as ff:
            n = len(res)
            m = len(f_list)
            print('n : %d, m : %d' % (n, m))
            for i in range(n):
                print(f_list[i])
                print(self.dict[0][res[i]])
                #print(type(f_list[i]))
                #print(type(res[i]))
                ff.write(f_list[i] + '\t' + self.dict[0][res[i]])


    def work(self):
        for fol in range(2, 5):
            self.prepare(fol)
            self.create_mfcc()
            self.learn(fol)

a = Learner()
#a.work()
a.predict()
#a.evaluation()
