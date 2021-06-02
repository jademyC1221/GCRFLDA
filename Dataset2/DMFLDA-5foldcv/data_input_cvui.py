# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 11:37:00 2020

@author: chenmeijun
"""

''' cite this:
M. Zeng et al., "DMFLDA: A deep learning framework for predicting IncRNA–disease associations," 
  in IEEE/ACM Transactions on Computational Biology and Bioinformatics, doi: 10.1109/TCBB.2020.2983958.'''


# encoding=utf-8
import random
from scipy.io import loadmat
import pickle
import numpy as np
from  hyperparams import Hyperparams as params

random.seed(params.static_random_seed)
neg_pos_ratio = params.neg_pos_ratio  #1
train_val_ratio = params.train_val_ratio  #0.9


class DataLoader:
    def __init__(self,pos_set,neg_set,postest1,postest2,postest3,postest4,postest5,\
                           negtrain1,negtrain2,negtrain3,negtrain4,negtrain5,\
                               postrain1,postrain2,postrain3,postrain4,postrain5):
        # with open('../data_processing/data.pkl', 'rb') as file:
        #     pos_set, neg_set = pickle.load(file)
        #     import pickle
        # with open('../data_processing/matrix.npy', 'rb') as file:
        #     matrix = np.load(file)
        m = loadmat("./data_processing/interMatrix.mat")
        matrix = m['interMatrix']

        self.matrix = matrix
        self.pos_set = pos_set
        self.neg_set = neg_set
        self.pos_size = len(pos_set)
        self.train_set = self.pos_set + self.neg_set  # initializer

    def coor_to_sample(self, batch):
        XL_batch = []
        XR_batch = []
        Y_batch = []
        for i, j, l in batch:
            temp = self.matrix[i][j]
            self.matrix[i][j] = 0
            XL_batch.append(self.matrix[i])
            XR_batch.append(self.matrix[:, j])
            self.matrix[i][j] = temp
            Y_batch.append(l)
        XL_batch = np.array(XL_batch)
        XR_batch = np.array(XR_batch)
        Y_batch = np.array(Y_batch).reshape((-1, 1))
        return XL_batch, XR_batch, Y_batch

    def shuffle(self):
        random.shuffle(self.train_set)

    # def leave_one_out(self, id):
    #     assert id >= 0 and id <= len(self.pos_set)

    #     neg_size = (self.pos_size - 1) * neg_pos_ratio
    #     neg_set = self.neg_set
    #     random.shuffle(neg_set)
    #     neg_set = neg_set[:neg_size]

    #     train_set = neg_set + self.pos_set[:id] + self.pos_set[id:]
    #     self.train_set = train_set
    #     self.train_size = len(train_set)
    #     self.val_set = [self.pos_set[id]]
    #     self.val_size = 1
    
    def fivefold(self, id):
    
        if id == 0:
            train_set = self.negtrain1 + self.postrain1
            self.train_set = train_set
            self.train_size = len(train_set)
            self.val_set = postest1
            self.val_size = len(postest1)
            
        elif id == 1:
            train_set = self.negtrain2 + self.postrain2
            self.train_set = train_set
            self.train_size = len(train_set)
            self.val_set = postest2
            self.val_size = len(postest2)
            
        elif id ==2:
            train_set = self.negtrain3 + self.postrain3
            self.train_set = train_set
            self.train_size = len(train_set)
            self.val_set = postest3
            self.val_size = len(postest3)
            
        elif id == 3:
            train_set = self.negtrain4 + self.postrain4
            self.train_set = train_set
            self.train_size = len(train_set)
            self.val_set = postest4
            self.val_size = len(postest4)
            
        else:
            train_set = self.negtrain5 + self.postrain5
            self.train_set = train_set
            self.train_size = len(train_set)
            self.val_set = postest5
            self.val_size = len(postest5)

    def sample_a_col(self, col_id):
        cols = []
        for i, x in enumerate(self.matrix[:, col_id]):
            cols.append((i, col_id, x))
        return cols


if __name__ == '__main__':
    dl = DataLoader()
    # print(dl.train_set)
    dl.shuffle()







