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

    print(line_list[0])
    print(len(line_list))

    num_valid = len(line_list) * 0.2

    for line in line_list:
        parts = line.split('\t')
        file_list.append(path + parts[0])

        if (parts[1] not in dict_label):
            dict_label[parts[1]] = num_label
            num_label = num_label + 1
        label_list.append(dict_label[parts[1]])

    print(num_label)

    if (overwrite):
        data_1 = []
        data_2 = []
        data_3 = []
        num = 0
        for item in file_list:
            #print(item)
            y, sr=soundfile.read(item)
            y = np.mean(y.T, axis=0)

            D_1 = np.abs(librosa.stft(y, n_fft = 512)) ** 2
            D_2 = np.abs(librosa.stft(y, n_fft = 2048)) ** 2
            D_3 = np.abs(librosa.stft(y, n_fft = 8192)) ** 2
            
            S_1 = librosa.feature.melspectrogram(S = D_1)
            S_2 = librosa.feature.melspectrogram(S = D_2)
            S_3 = librosa.feature.melspectrogram(S = D_3)
            #print(S.shape)

            S_1 = librosa.feature.mfcc(S=librosa.power_to_db(S_1), n_mfcc = 64)
            S_2 = librosa.feature.mfcc(S=librosa.power_to_db(S_2), n_mfcc = 64)
            S_3 = librosa.feature.mfcc(S=librosa.power_to_db(S_3), n_mfcc = 64)
            #print(S.shape)
            #time.sleep(1000)

            data_1.append(S_1.T)
            data_2.append(S_2.T)
            data_3.append(S_3.T)

            if (num % 100 == 0):
                print(num)
            num = num + 1

        data_1 = np.array(data_1)
        data_2 = np.array(data_2)
        data_3 = np.array(data_3)
        print(data_1.shape)
        print(data_2.shape)
        print(data_3.shape)

        f_1 = file('data_mfcc_1024.npy', 'w')
        f_2 = file('data_mfcc_2048.npy', 'w')
        f_3 = file('data_mfcc_4096.npy', 'w')
        np.save(f_1, data_1)
        np.save(f_2, data_2)
        np.save(f_3, data_3)
        f_1.close()
        f_2.close()
        f_3.close()
        print(' Data End ')

prepare_mfcc()
