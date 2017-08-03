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

def Load_list(name):
    f = open(name, 'r')
    load_name_list = []
    for line in open(name):
        line = f.readline()
        load_name_list.append(line)
    f.close()

def save_list(data, name):
    f = open(name, 'w')
    for item in data:
        f.write('%s' % item)
    f.close()

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
    name_list = []
    dict_label = {}
    dict_name = {}
    num_label = 0
    num_name = 0
    num_num = 0

    with open(meta_path, 'r') as ff:
        for line in ff:
            line_list.append(line)
            parts = line.split('\t')
            name = parts[2]
            #print(name)
            if (name not in dict_name):
                name_list.append(name)
                dict_name[name] = 0
                num_name = num_name + 1
            dict_name[name] = dict_name[name] + 1
            num_num = num_num + 1

    print(num_name)
    #time.sleep(1000)
    random.shuffle(name_list)
    save_list(name_list, 'name_list.txt')

    for i in range(5):
        name = name_list[i]
        print(name)
        for line in line_list:
            parts = line.split('\t')
            if (parts[2] == name):
                print(parts[1])
                break
    #-----------------------------

    print('num_list : %d' % len(name_list))

    num_train_name = int(len(name_list) * 0.8)

    print('num_train_name : %d' % num_train_name)

    num_train = 0

    for i in range(num_train_name):
        num_train = num_train + dict_name[name_list[i]]
    print('num_train : %d' % num_train)
    print('total_num : %d' % num_num)
    #time.sleep(1000)
    #--------------------------------------

    for name in name_list:
        for line in line_list:
            parts = line.split('\t')
            if (parts[2] != name):
                continue

            file_list.append(path + parts[0])

            if (parts[1] not in dict_label):
                dict_label[parts[1]] = num_label
                num_label = num_label + 1
            label_list.append(dict_label[parts[1]])

    print('num_label : %d' % num_label)

    label = np.array(label_list)
    print(label.shape)

    Save(label, 'label', num_train)
    data_2048 = []
    data_8196 = []
    data_cqt = []
    data_mel = []
    num = 0
    for item in file_list:
        y, sr=soundfile.read(item)
        y = np.mean(y.T, axis=0)

        D_2 = np.abs(librosa.stft(y, n_fft = 2048)) ** 2
        D_3 = np.abs(librosa.stft(y, n_fft = 8192)) ** 2
        
        S_2 = librosa.feature.melspectrogram(S = D_2)
        S_3 = librosa.feature.melspectrogram(S = D_3)

        data_mel.append(librosa.power_to_db(S_2).T)

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
    data_mel = np.array(data_mel)
    print(data_2048.shape)
    print(data_8196.shape)
    print(data_cqt.shape)
    print(data_mel.shape)

    Save(data_2048, 'data_2048', num_train)
    Save(data_8196, 'data_8192', num_train)
    Save(data_cqt, 'data_cqt', num_train)
    Save(data_mel, 'data_mel', num_train)
    
    print(' Data End ')

def Random():
    meta_path = path + 'meta.txt'
    file_list = []
    line_list = []
    label_list = []
    name_list = []
    dict_label = {}
    dict_name = {}
    num_label = 0
    num_name = 0
    num_num = 0

    with open(meta_path, 'r') as ff:
        for line in ff:
            line_list.append(line)
            parts = line.split('\t')
            name = parts[2]

            if (name not in dict_name):
                dict_name[name] = 0
                num_name = num_name + 1
            dict_name[name] = dict_name[name] + 1
            num_num = num_num + 1

    print(num_name)

    load_name_list('name_list.txt')

    print('num_list : %d' % len(name_list))

    num_train_name = int(len(name_list) * 0.8)

    print('num_train_name : %d' % num_train_name)

    num_train = 0

    for i in range(num_train_name):
        num_train = num_train + dict_name[load_name_list[i]]
    print('num_train : %d' % num_train)
    print('total_num : %d' % num_num)

    #--------------------------------------

    iid = {}
    for name in load_name_list:
        for line in line_list:
            parts = line.split('\t')
            if (parts[2] != name):
                continue

            if (parts[1] not in dict_label):
                iid[parts[2]] = parts[1]
                dict_label[parts[1]] = num_label
                num_label = num_label + 1

    Satistics = np.zeros(15)
    for i in range(num_train_name):
        Id = iid[load_name_list[i]]
        Satistics[Id] = Satistics[Id] + dict_name[load_name_list[i]]

    Total = np.zeros(15)
    for i in range(num_train_name):
        Id = iid[load_name_list[i]]
        Total[Id] = Total[Id] + dict_name[load_name_list[i]]

    for i in range(15):
        valid = Total[i] - Satistics[i]
        print('total : %d, train : %d, valid : %d, valid_ratio : %lf' % (Total[i], Satistics[i], valid, 1. * valid / Total[i]))

#prepare_mfcc()
Random()