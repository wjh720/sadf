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

Type = 'development'
path = '../data/TUT-acoustic-scenes-2017-' + Type + '/'
overwrite = True

def prepare_data():
    meta_path = path + 'meta.txt'
    file_list = []
    label_list = []
    dict_label = {}
    num_label = 0

    with open(meta_path, 'r') as ff:
        for line in ff:
            #print(line)
            parts = line.split('\t')
            #print(parts)
            #time.sleep(100)
            file_list.append(path + parts[0])

            if (parts[1] not in dict_label):
                dict_label[parts[1]] = num_label
                num_label = num_label + 1
            label_list.append(dict_label[parts[1]])

    print(num_label)

    if (overwrite):
        label = np.array(label_list)
        print(label.shape)
        f = file('label.npy', 'w')
        np.save(f, label)
        f.close()
        print(' Label End ')

        data = []
        num = 0
        for item in file_list:
            #print(item)
            y, sr=soundfile.read(item)
            y = np.mean(y.T, axis=0)

            D = np.abs(librosa.stft(y)) ** 2
            #print(D.shape)
            S = librosa.feature.melspectrogram(S = D)
            #print(S.shape)

            S = librosa.power_to_db(S)
            #print(S.shape)
            #time.sleep(1000)

            data.append(S.T)

            if (num % 100 == 0):
                print(num)
            num = num + 1

        data = np.array(data)
        print(data.shape)

        f = file('data.npy', 'w')
        np.save(f, data)
        f.close()
        print(' Data End ')

def prepare_mfcc():
    meta_path = path + 'meta.txt'
    file_list = []
    label_list = []
    dict_label = {}
    num_label = 0

    with open(meta_path, 'r') as ff:
        for line in ff:
            #print(line)
            parts = line.split('\t')
            #print(parts)
            #time.sleep(100)
            file_list.append(path + parts[0])

            if (parts[1] not in dict_label):
                dict_label[parts[1]] = num_label
                num_label = num_label + 1
            label_list.append(dict_label[parts[1]])

    print(num_label)

    if (overwrite):
        data = []
        num = 0
        for item in file_list:
            #print(item)
            y, sr=soundfile.read(item)
            y = np.mean(y.T, axis=0)

            D = np.abs(librosa.stft(y)) ** 2
            #print(D.shape)
            S = librosa.feature.melspectrogram(S = D)
            #print(S.shape)

            S = librosa.feature.mfcc(S=librosa.power_to_db(S))
            #print(S.shape)
            #time.sleep(1000)

            data.append(S.T)

            if (num % 100 == 0):
                print(num)
            num = num + 1

        data = np.array(data)
        print(data.shape)

        f = file('data_mfcc.npy', 'w')
        np.save(f, data)
        f.close()
        print(' Data End ')

#prepare_data()
prepare_mfcc()