#!/usr/bin/env python2
import numpy
from features import sigproc
from features import mfcc
import scipy.io.wavfile as wav
import vad
import os
from six.moves import cPickle as pickle



def feature_extract(wav_name, winlen=0.025, winstep=0.01):
    """This function returns (mfcc) feature vectors extracted from wav_name"""
    rate, signal = wav.read(wav_name)
    signal = numpy.sum(signal, axis=1)/signal.shape[1]
    signal = sigproc.framesig(signal, rate*winlen, rate*winstep)
    signal = vad.vad_filter(signal)
    signal = sigproc.deframesig(signal, 0, rate*winlen, rate*winstep)
    mfcc_feat = mfcc(signal, rate)
    return mfcc_feat

def model_preprocess(directory_name):
    """This function extracts *.wav files from each sub directory, store all features to
    train_dataset (test_dataset), labels to train_labels (test_labels)"""

    """ data_folders is the list of sub folders for the training data"""
    data_folders = [
        os.path.join(directory_name, d) for d in sorted(os.listdir(directory_name))
        if os.path.isdir(os.path.join(directory_name, d))]
    num_classes = len(data_folders)
    len_feature = 13
    train_dataset = numpy.ndarray(shape = (0, len_feature), dtype = numpy.float64)
    train_labels = numpy.ndarray(shape = (0), dtype=numpy.int)
    class_names = []
    for folder in data_folders:
        class_names.append(os.path.split(folder)[1])
        for wav in os.listdir(folder):
            wav_file = os.path.join(folder, wav)
            if os.path.exists(wav_file) and os.path.splitext(wav_file)[1]=='.wav':
                temp_feat = feature_extract(wav_file)
                temp_labels = class_names.index(os.path.split(folder)[1])*numpy.ones(shape = (temp_feat.shape[0]), dtype=numpy.int)
                train_dataset = numpy.concatenate((train_dataset, temp_feat), axis = 0)
                train_labels = numpy.concatenate((train_labels, temp_labels), axis = 0)
    return class_names, train_dataset, train_labels

def model_save(class_names, train_dataset, train_labels):
    pickle_file = 'train_data'
    save = {
        'class_names': class_names,
        'train_dataset': train_dataset,
        'train_labels': train_labels,
    }
    f = open(pickle_file, 'wb')
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()

def model_load(pickle_file):
    f = open(pickle_file, 'r')
    save = pickle.load(f)
    return save['class_names'], save['train_dataset'], save['train_labels']
