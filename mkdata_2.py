import time
import os
import sys
import logging
import random
import numpy as np
import copy
import importlib
import librosa
import soundfile
import collections

Type = 'development'
path = '../data/TUT-acoustic-scenes-2017-' + Type + '/'
overwrite = True

def Save(data, name, train):
    f = file(name + 'train', 'w')
    np.save(f, data[:train])
    f.close()

    f = file(name + 'valid', 'w')
    np.save(f, data[train:])
    f.close()

def prepare_mfcc():
    meta_path = path + 'meta.txt'
    file_list = []
    line_list = []
    label_list = []
    dict_label = {}
    num_label = 0

    with open(meta_path, 'r') as ff:
        for line in ff:
            line_list.append(line)

    random.shuffle(line_list)

    print('num_list : %d' % len(line_list))

    num_train = int(len(line_list) * 0.8)

    print('num_train : %d' % num_train)

    for line in line_list:
        parts = line.split('\t')
        file_list.append(path + parts[0])

        if (parts[1] not in dict_label):
            dict_label[parts[1]] = num_label
            num_label = num_label + 1
        label_list.append(dict_label[parts[1]])

    print(num_label)

    label = np.array(label_list)
    print(label.shape)

    Save(label, 'label', num_train)
    data_2048 = []
    data_8196 = []
    data_cqt = []
    num = 0
    for item in file_list:
        y, sr=soundfile.read(item)
        y = np.mean(y.T, axis=0)

        D_2 = np.abs(librosa.stft(y, n_fft = 2048)) ** 2
        D_3 = np.abs(librosa.stft(y, n_fft = 8192)) ** 2
        
        S_2 = librosa.feature.melspectrogram(S = D_2)
        S_3 = librosa.feature.melspectrogram(S = D_3)

        S_2 = librosa.feature.mfcc(S=librosa.power_to_db(S_2), n_mfcc = 64)
        S_3 = librosa.feature.mfcc(S=librosa.power_to_db(S_3), n_mfcc = 64)

        data_2048.append(S_2.T)
        data_8196.append(S_3.T)

        chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma = 64)
        data_cqt.append(chroma_cq.T)

        if (num % 100 == 0):
            print(num)
        num = num + 1

    data_2048 = np.array(data_2048)
    data_8196 = np.array(data_8196)
    data_cqt = np.array(data_cqt)
    print(data_2048.shape)
    print(data_8196.shape)
    print(data_cqt.shape)

    Save(data_2048, 'data_2048', num_train)
    Save(data_8196, 'data_8192', num_train)
    Save(data_cqt, 'data_cqt', num_train)
    
    print(' Data End ')

prepare_mfcc()
