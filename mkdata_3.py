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

path = '../data/TUT-acoustic-scenes-2017-development/'

def Save(data, name):
    f = file(name, 'w')
    np.save(f, data)
    f.close()

def Work(name, save_name):
    dict_label = {}
    num_label = 0
    label_list = []
    file_list = []
    with open(name, 'r') as ff:
        for line in ff:
            parts = line.split('\t')

            file_list.append(path + parts[0])

            if (parts[1] not in dict_label):
                dict_label[parts[1]] = num_label
                num_label = num_label + 1
            label_list.append(dict_label[parts[1]])

    label = np.array(label_list)
    print(label.shape)

    Save(label, save_name + '_label')

    data_2048 = []
    data_4096 = []
    data_8192 = []
    data_cqt = []
    data_mel = []
    num = 0
    for item in file_list:
        y, sr=soundfile.read(item)
        y = np.mean(y.T, axis=0)

        D_1 = np.abs(librosa.stft(y, n_fft = 4096)) ** 2
        D_2 = np.abs(librosa.stft(y, n_fft = 2048)) ** 2
        D_3 = np.abs(librosa.stft(y, n_fft = 8192)) ** 2
        
        S_1 = librosa.feature.melspectrogram(S = D_1)
        S_2 = librosa.feature.melspectrogram(S = D_2)
        S_3 = librosa.feature.melspectrogram(S = D_3)

        data_mel.append(librosa.power_to_db(S_2).T)

        S_1 = librosa.feature.mfcc(S=librosa.power_to_db(S_1), n_mfcc = 64)
        S_2 = librosa.feature.mfcc(S=librosa.power_to_db(S_2), n_mfcc = 64)
        S_3 = librosa.feature.mfcc(S=librosa.power_to_db(S_3), n_mfcc = 64)

        data_2048.append(S_2.T)
        data_4096.append(S_1.T)
        data_8192.append(S_3.T)

        chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma = 64)
        data_cqt.append(chroma_cq.T)

        if (num % 100 == 0):
            print(num)
        num = num + 1

    data_2048 = np.array(data_2048)
    data_4096 = np.array(data_4096)
    data_8192 = np.array(data_8192)
    data_cqt = np.array(data_cqt)
    data_mel = np.array(data_mel)
    print(data_2048.shape)
    print(data_4096.shape)
    print(data_8192.shape)
    print(data_cqt.shape)
    print(data_mel.shape)

    Save(data_2048, save_name + '_data_2048')
    Save(data_4096, save_name + '_data_4096')
    Save(data_8192, save_name + '_data_8192')
    Save(data_cqt, save_name + '_data_cqt')
    Save(data_mel, save_name + '_data_mel')
    
    print(' Data End '+ save_name)

def prepare_mfcc():
    meta_path = path + 'evaluation_setup/'
    for fol in range(1, 5):
        fold_name = meta_path + ('fold%d' % fol) 
        Work(fold_name + '_train.txt', fold_name + '_train')
        Work(fold_name + '_evaluate.txt', fold_name + '_evaluate')

def prepare_evaluation():
    path = '../data/TUT-acoustic-scenes-2017-evaluation/'
    meta_path = path + 'evaluation_setup/'
    for fol in range(1, 5):
        fold_name = meta_path + 'text.txt'
        Work(fold_name, fold_name + '_test')


#prepare_mfcc()
prepare_evaluation()