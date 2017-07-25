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

    def log_model_summary(self):
        """Prints model summary to the logging interface.

        Similar to Keras model summary
        """

        layer_name_map = {
            'BatchNormalization': 'BatchNorm',
        }
        import keras.backend as keras_backend

        self.logger.debug('  Model summary')
        self.logger.debug(
            '    {type:<15s} | {out:20s} | {param:6s}  | {name:21s}  | {conn:27s} | {act:7s} | {init:7s}'.format(
                type='Layer type',
                out='Output',
                param='Param',
                name='Name',
                conn='Connected to',
                act='Activ.',
                init='Init')
        )

        self.logger.debug(
            '    {type:<15s} + {out:20s} + {param:6s}  + {name:21s}  + {conn:27s} + {act:7s} + {init:6s}'.format(
                type='-' * 15,
                out='-' * 20,
                param='-' * 6,
                name='-' * 21,
                conn='-' * 27,
                act='-' * 7,
                init='-' * 6)
        )

        for layer in self.model.layers:
            connections = []
            for node_index, node in enumerate(layer.inbound_nodes):
                for i in range(len(node.inbound_layers)):
                    inbound_layer = node.inbound_layers[i].name
                    inbound_node_index = node.node_indices[i]
                    inbound_tensor_index = node.tensor_indices[i]
                    connections.append(inbound_layer + '[' + str(inbound_node_index) +
                                       '][' + str(inbound_tensor_index) + ']')

            config = DottedDict(layer.get_config())
            layer_name = layer.__class__.__name__
            if layer_name in layer_name_map:
                layer_name = layer_name_map[layer_name]

            if config.get_path('kernel_initializer.class_name') == 'VarianceScaling':
                init = str(config.get_path('kernel_initializer.config.distribution', '---'))
            elif config.get_path('kernel_initializer.class_name') == 'RandomUniform':
                init = 'uniform'
            else:
                init = '---'

            self.logger.debug(
                '    {type:<15s} | {shape:20s} | {params:6s}  | {name:21s}  | {connected:27s} | {activation:7s} | {init:7s}'.format(
                    type=layer_name,
                    shape=str(layer.output_shape),
                    params=str(layer.count_params()),
                    name=str(layer.name),
                    connected=str(connections[0]) if len(connections) > 0 else '---',
                    activation=str(config.get('activation', '---')),
                    init=init,

                )
            )

        trainable_count = int(
            numpy.sum([keras_backend.count_params(p) for p in set(self.model.trainable_weights)])
        )

        non_trainable_count = int(
            numpy.sum([keras_backend.count_params(p) for p in set(self.model.non_trainable_weights)])
        )

        self.logger.debug('  ')
        self.logger.debug('  Parameters')
        self.logger.debug('    Trainable\t[{param_count:,}]'.format(param_count=int(trainable_count)))
        self.logger.debug('    Non-Trainable\t[{param_count:,}]'.format(param_count=int(non_trainable_count)))
        self.logger.debug(
            '    Total\t\t[{param_count:,}]'.format(param_count=int(trainable_count + non_trainable_count)))
        self.logger.debug('  ')

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

        def lam(X):
            print("xxxxxxxxxxxxxxxxxxxxxxxxxx")
            print(X.shape)
            X=K.max(X,axis=2)
            print(X.shape)
            return X

        self.model = Sequential()

        self.model.add(Reshape((86, 128, 1), input_shape=(86, 128)))
        self.model.add(Conv2D(32, (3, 3), padding='same',activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(32, (3, 3), padding='same',activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size = (2, 2)))
        self.model.add(Dropout(0.1))

        self.model.add(Conv2D(32, (3, 3), padding='same',activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(32, (3, 3), padding='same',activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size = (2, 2)))
        self.model.add(Dropout(0.2))

        self.model.add(Conv2D(32, (3, 3), padding='same',activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(32, (3, 3), padding='same',activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size = (10, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Lambda(lam,output_shape=(1,32)))
        self.model.add(Flatten())
        self.model.add(Dense(15, activation='softmax'))
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=["accuracy"])


    def work(self):
        self.prepare()
        self.create_model()
        #self.log_model_summary()
        self.learn()

a = Learner()
a.work()


