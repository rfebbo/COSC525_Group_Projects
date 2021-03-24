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

    train_labels = pd.read_csv('project3_COSC525/fairface_label_train.csv')
    val_labels = pd.read_csv('project3_COSC525/fairface_label_val.csv')

    age_classes = train_labels.age.unique()
    gender_classes = train_labels.gender.unique()
    race_classes = train_labels.race.unique()

    #training labels
    codes, _ = train_labels.age.factorize()
    age_t_labels = np.zeros((len(codes), age_classes.size))
    for i, c in enumerate(codes):
        age_t_labels[i,c] = 1

    codes, _ = train_labels.gender.factorize()
    gender_t_labels = np.zeros((len(codes), gender_classes.size))
    for i, c in enumerate(codes):
        gender_t_labels[i,c] = 1
    
    codes, _ = train_labels.race.factorize()
    race_t_labels = np.zeros((len(codes), race_classes.size))
    for i, c in enumerate(codes):
        race_t_labels[i,c] = 1

    #validation labels
    codes, _ = val_labels.age.factorize()
    age_v_labels = np.zeros((len(codes), age_classes.size))
    for i, c in enumerate(codes):
        age_v_labels[i,c] = 1

    codes, _ = val_labels.gender.factorize()
    gender_v_labels = np.zeros((len(codes), gender_classes.size))
    for i, c in enumerate(codes):
        gender_v_labels[i,c] = 1
    
    codes, _ = val_labels.race.factorize()
    race_v_labels = np.zeros((len(codes), race_classes.size))
    for i, c in enumerate(codes):
        race_v_labels[i,c] = 1


    dataset = {'train_norm' : train_norm, 
    'val_norm' : val_norm, 
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