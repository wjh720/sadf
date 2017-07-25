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

def prepare_data():
    meta_path = path + 'meta.txt'
    file_list = []
    label_list = []
    dict_label = {}
    num_label = 0

    with open(meta_path, 'r') as ff:
        for line in ff:
            parts = line.split(' ')
            file_list.append(parts[0])

            if (parts[1] not in dict_label):
                dict_label[parts[1]] = num_label
                num_label = num_label + 1
            label_list.append(dict_label[parts[1]])

    print(num_label)
    label = np.array(label_list)

    data = []

    for file in file_list:
        y, sr=soundfile.read(item)
        y = np.mean(y.T, axis=0)

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

        print(S.shape)

        data.append(S)

    data = data.array(data)

    print(data.shape)

    f = file('data.npy', 'w')
    np.save(f, data)
    f.close()

prepare_data()