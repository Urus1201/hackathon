from config import PARAMS
from utils import map_label

import numpy as np
import pandas as pd

import torch
from sklearn import preprocessing
from imblearn.over_sampling import RandomOverSampler

opt = PARAMS()

class DATA_LOADER_HK(object):
  def __init__(self, opt, ):
    #Load the training dataset. You need to copy it to Google drive. and place in the DS_Hackathon2023 directory.
    train_feature = pd.read_csv('../data/train/train_feature.csv').values
    train_label = pd.read_csv('../data/train/train_label.csv').values.flatten()
    train_attribute = pd.read_csv('../data/train/train_attribute.csv').values
    train_image_names = pd.read_csv('../data/train/train_image_names.csv').values

    train_classes = np.unique(train_label)

    #Create a validation data out of training data
    num_val_classes = int(0.4*len(train_classes))
    num_train_classes = len(train_classes) - num_val_classes

    self.train_classes = train_classes[:num_train_classes]
    self.val_classes = train_classes[num_train_classes:]
    self.nall_classes = len(self.train_classes) + len(self.val_classes)

    ros = RandomOverSampler()
    train_feature, train_label = ros.fit_resample(train_feature, train_label)

    self.allattribute = train_attribute

    #create new train validation split for labels and features
    new_train_loc = []
    for cl in self.train_classes:
      new_train_loc.append((np.nonzero(train_label == cl)[0]).tolist())

    new_train_loc = [item for sublist in new_train_loc for item in sublist]

    new_val_loc = []
    for cl in self.val_classes:
      new_val_loc.append((np.nonzero(train_label == cl)[0]).tolist())

    new_val_loc = [item for sublist in new_val_loc for item in sublist]

    self.train_label = train_label[new_train_loc]
    self.train_feature = train_feature[new_train_loc]
    self.val_label = train_label[new_val_loc]
    self.val_feature = train_feature[new_val_loc]

    print('Train classes', len(self.train_classes), self.train_classes)
    print('Val classes', len(self.val_classes), self.val_classes)

    print('Number of train samples', len(self.train_label))
    print('Number of Val samples', len(self.val_label))
    print('attribute size ', self.allattribute.shape)

    scaler = preprocessing.StandardScaler()

    _train_feature = scaler.fit_transform(self.train_feature)
    _val_feature = scaler.transform(self.val_feature)

    self.train_feature = torch.from_numpy(_train_feature).float()
    self.train_label = torch.from_numpy(self.train_label).long()
    self.val_label = torch.from_numpy(self.val_label).long()
    self.val_feature = torch.from_numpy(_val_feature).float()
    self.allattribute = torch.from_numpy(self.allattribute).float()
    self.ntrain = self.train_feature.size()[0]

  def next_batch(self, batch_size):
    idx = torch.randperm(self.ntrain)[0:batch_size]
    batch_feature = self.train_feature[idx]
    batch_label = self.train_label[idx]
    mapped_batch_label = map_label(batch_label, self.train_classes)
    batch_att = self.allattribute[mapped_batch_label]
    return batch_feature, batch_att