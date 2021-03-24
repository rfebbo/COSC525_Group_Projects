import numpy as np
import os
from PIL import Image
import pandas as pd

import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def read_imgs(path):
    files = os.listdir(path)
    files.sort(key=natural_keys)

    images = []
    for file in files:
        pix = np.array(Image.open(path + file))
        images.append(pix)

    return np.asarray(images)

def encode(labels):

    codes, _ = labels.factorize()
    encoded_labels = np.zeros((len(codes), len(labels.unique())))
    for i, c in enumerate(codes):
        encoded_labels[i,c] = 1

    return encoded_labels

def read_data():
    
    train = read_imgs('./project3_COSC525/train/')
    val = read_imgs('./project3_COSC525/val/')

    train_max = np.amax(train, axis=0)
    train_min = np.amin(train, axis=0)

    #normalize
    train_norm = (train - train_min) / (train_max - train_min)
    val_norm = (val - train_min) / (train_max - train_min)

    train_norm = train_norm.reshape(-1,32,32,1)
    val_norm = val_norm.reshape(-1,32,32,1)

    #get labels
    train_labels = pd.read_csv('project3_COSC525/fairface_label_train.csv')
    val_labels = pd.read_csv('project3_COSC525/fairface_label_val.csv')

    #get label labels
    age_classes = train_labels.age.unique()
    gender_classes = train_labels.gender.unique()
    race_classes = train_labels.race.unique()

    #encode training labels
    age_t_labels = encode(train_labels.age)
    gender_t_labels = encode(train_labels.gender)
    race_t_labels = encode(train_labels.race)

    #encode validation labels
    age_v_labels = encode(val_labels.age)
    gender_v_labels = encode(val_labels.gender)
    race_v_labels = encode(val_labels.race)

    dataset = {
        'train' : train_norm, 
        'val' : val_norm, 
        'age_t_labels' : age_t_labels, 
        'gender_t_labels' : gender_t_labels, 
        'race_t_labels' : race_t_labels, 
        'age_v_labels' : age_v_labels,  
        'gender_v_labels' : gender_v_labels,  
        'race_v_labels' : race_v_labels, 
        'age_classes' : age_classes, 
        'gender_classes' : gender_classes, 
        'race_classes' : race_classes}

    return dataset