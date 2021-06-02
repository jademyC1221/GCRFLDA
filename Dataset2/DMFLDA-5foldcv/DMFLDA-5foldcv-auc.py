# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 21:11:03 2021

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
neg_pos_ratio = params.neg_pos_ratio
train_val_ratio = params.train_val_ratio

import tensorflow as tf
from sklearn import metrics
import numpy as np

import data_input_cvui as data_input
import DMF_model_cvui as DMF_model
from  hyperparams import Hyperparams as params


batch_size = params.batch_size
epoch_num = params.epoch_num
tf.set_random_seed(params.tf_random_seed)


def validate(data_set, model, sess, roc_params=False):
    # XL_batch, XR_batch, Y_batch = dl.coor_to_sample(data_set)
    XL_batch, XR_batch, Y_batch = coor_to_sample(data_set)
    y_pred, y_score, accuracy, loss = sess.run(
        [model.prediction, model.score, model.accuracy, model.loss],
        feed_dict={model.XL_input: XL_batch, model.XR_input: XR_batch,
                   model.Y_input: Y_batch})
    if not roc_params:
        return Y_batch, y_pred, y_score, accuracy, loss
    else:
        fpr, tpr, thresholds = metrics.roc_curve(Y_batch, y_score)
        return Y_batch, y_pred, y_score, accuracy, loss, fpr, tpr


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def coor_to_sample(batch):
    XL_batch = []
    XR_batch = []
    Y_batch = []
    for i, j, l in batch:
        temp = matrix[i][j]
        matrix[i][j] = 0
        XL_batch.append(matrix[i])
        XR_batch.append(matrix[:, j])
        matrix[i][j] = temp
        Y_batch.append(l)
    XL_batch = np.array(XL_batch)
    XR_batch = np.array(XR_batch)
    Y_batch = np.array(Y_batch).reshape((-1, 1))
    return XL_batch, XR_batch, Y_batch

def cross_validation(train_size,train_set,val_set):
    model = DMF_model.DMF()
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        print('begin training:')
        for epoch in range(epoch_num):

            print(('epoch %d' % epoch).center(50, '='))
            ys_pred = []
            ys_true = []
            ys_score = []
            for iter, indices in enumerate(range(0, train_size, batch_size)):
                X_coor_batch = train_set[indices:indices + batch_size]
                XL_batch, XR_batch, Y_batch = coor_to_sample(X_coor_batch)
                y_pred, y_score, loss, reg_loss, _ = sess.run(
                    [model.prediction, model.score, model.loss, model.reg_loss, model.optimizer],
                    feed_dict={model.XL_input: XL_batch, model.XR_input: XR_batch,
                               model.Y_input: Y_batch})
                ys_pred.append(y_pred)
                ys_score.append(y_score)
                ys_true.append(Y_batch)

            ys_true = np.concatenate(ys_true, 0) #按列拼接
            ys_pred = np.concatenate(ys_pred, 0)
            ps = metrics.precision_score(ys_true, ys_pred)
            rs = metrics.recall_score(ys_true, ys_pred)
            f1 = metrics.f1_score(ys_true, ys_pred)
            accuracy = metrics.accuracy_score(ys_true, ys_pred)
            print(
                ('Training ACC:%.3f, LOSS:%.3f, Precision:%.3f, Recall:%.3f,F1:%.3f' % (
                    accuracy, loss, ps, rs, f1)).center(
                    50, '='))
        # leave one validation
        # y_true, y_pred, y_score, accuracy, loss = validate(dl.val_set, model, sess)
        
        # print(
        #     ('Leave-One-Validation ACC:%.3f, LOSS:%.3f RETURN:%d, SCORE:%.2f' % (
        #         accuracy, loss, y_pred, y_score)).center(
        #         50, '#'))
        
        # print(
        #     ('Leave-One-Validation ACC:%.3f, LOSS:%.3f , SCORE:%.2f' % (
        #         accuracy, loss, y_score)).center(
        #         50, '#'))
        # column validation
        # row_id = dl.val_set[0][0]
        # col_id = dl.val_set[0][1]
        # x_val = dl.sample_a_col(col_id)
        x_val = val_set
        y_true, y_pred, y_score, accuracy, loss, fpr, tpr = validate(x_val, model, sess, roc_params=True)
    # destroy model
    tf.reset_default_graph()
    return fpr, tpr, accuracy, y_true, y_pred, y_score



pkl_file=open('./data_processing/data1227.pkl','rb')
pos_set =pickle.load(pkl_file)
neg_set =pickle.load(pkl_file)
pkl_file.close()

m = loadmat("./data_processing/interMatrix.mat")
matrix = m['interMatrix']

k=5
random.shuffle(pos_set)
pos_size = len(pos_set)
num = int(pos_size / k)
# print(kf)
postest1 = pos_set[0:num]
postest2 = pos_set[num:num*2]
postest3 = pos_set[num*2:num*3]
postest4 = pos_set[num*3:num*4]
postest5 = pos_set[num*4:len(pos_set)]

postrain1 = pos_set[num:len(pos_set)]
postrain2 = pos_set[0:num] + pos_set[num*2:len(pos_set)]
postrain3 = pos_set[0:num*2] + pos_set[num*3:len(pos_set)]
postrain4 = pos_set[0:num*3] + pos_set[num*4:len(pos_set)]
postrain5 = pos_set[0:num*4]

random.shuffle(neg_set)
negtrain1 = neg_set[num:pos_size]
negtrain2 = neg_set[0:num] + neg_set[num*2:pos_size]
negtrain3 = neg_set[0:num*2] + neg_set[num*3:pos_size]
negtrain4 = neg_set[0:num*3] + neg_set[num*4:pos_size]
negtrain5 = neg_set[0:num*4]

# random.shuffle(neg_set)
# negtest1 = neg_set
# random.shuffle(neg_set)
# negtest2 = neg_set
# random.shuffle(neg_set)
# negtest3 = neg_set
# random.shuffle(neg_set)
# negtest4 = neg_set
# random.shuffle(neg_set)
# negtest5 = neg_set


# 保存矩阵每一列的预测结果
# ys = []
# xs = []
y_true_val = []
y_pred_val = []
y_prob_val = []
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
mean_fpr = np.linspace(0, 1, 100)
tprs=[]
aucs=[]
k=5

from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
mean_fpr = np.linspace(0, 1, 100)

tf.reset_default_graph()
for i in range(k):  # len(dl.pos_set)
    print('CROSS VALIDATION %d' % (i + 1))
    if i == 0:
        train_set = negtrain1 + postrain1
        train_set = train_set
        train_size = len(train_set)
        val_set = postest1
        val_size = len(val_set)
        
    elif i == 1:
        train_set = negtrain2 + postrain2
        train_set = train_set
        train_size = len(train_set)
        val_set = postest2
        val_size = len(val_set)
        
    elif i ==2:
        train_set = negtrain3 + postrain3
        train_set = train_set
        train_size = len(train_set)
        val_set = postest3
        val_size = len(val_set)
        
    elif i == 3:
        train_set = negtrain4 + postrain4
        train_set = train_set
        train_size = len(train_set)
        val_set = postest4
        val_size = len(val_set)
        
    else:
        train_set = negtrain5 + postrain5
        train_set = train_set
        train_size = len(train_set)
        val_set = postest5
        val_size = len(val_set)
    
    random.shuffle(train_set)
    random.shuffle(val_set)
    fpr, tpr, accuracy, y_true, y_pred, y_score = cross_validation(train_size,train_set,val_set)
    
    # row_id = dl.val_set[0][0]
    # col_id = dl.val_set[0][1]
    y_true_val.append(y_true)
    y_pred_val.append(y_pred)
    y_prob_val.append(y_score) 
    
    tprs.append(interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)

# # y is label matrix, 0 is test sample label, 1 is the label of positive samples, -1 is the lable of the negative samples
# xs = y_prob_val
# y = []
# for t in y_true_val:
#     if t == 0:
#         y.append(-1)
#     else:
#         y.append(0)
# ys=y
# xs = np.array(xs).transpose().squeeze(0)
# ys = np.array(ys).transpose()
# np.save('xs.npy', xs)
# np.save('ys.npy', ys)

# ps = metrics.precision_score(y_true_val, y_pred_val)
# rs = metrics.recall_score(y_true_val, y_pred_val)
# f1 = metrics.f1_score(y_true_val, y_pred_val)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)

from sklearn import metrics
y_real = np.concatenate(y_true_val)
y_proba = np.concatenate(y_prob_val)
mean_precision, mean_recall,_ = metrics.precision_recall_curve(y_real, y_proba)
mean_aupr = auc(mean_recall, mean_precision)

print('mean_auc: {:.4f}, mean_aupr: {:.4},'.format(mean_auc, mean_aupr))


# #保存
# import pickle
# output = open('DMFLDA_allneg_0.8982.pkl','wb')
# pickle.dump(tprs,output)
# pickle.dump(mean_fpr,output)
# pickle.dump(mean_tpr,output)
# pickle.dump(mean_auc,output)
# pickle.dump(y_real,output)
# pickle.dump(y_proba,output)
# pickle.dump(mean_aupr,output)
# pickle.dump(y_true_val,output)
# pickle.dump(y_pred_val,output)
# pickle.dump(y_prob_val,output)
# pickle.dump(tprs,output)
# output.close()

  #读取数据
import pickle
pkl_file=open('DMFLDA_allneg_0.8982.pkl','rb')
tprs = pickle.load(pkl_file)
mean_fpr = pickle.load(pkl_file)
mean_tpr = pickle.load(pkl_file)
mean_auc = pickle.load(pkl_file)
y_real = pickle.load(pkl_file)
y_proba = pickle.load(pkl_file)
mean_aupr = pickle.load(pkl_file)
y_true_val = pickle.load(pkl_file)
y_pred_val = pickle.load(pkl_file)
tprs = pickle.load(pkl_file)
pkl_file.close()


# import numpy as np
# plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
#         label='Chance', alpha=.8)
# plt.plot(mean_fpr, mean_tpr, color='b',
#         label=r'Mean ROC (AUC = %0.4f)' % (mean_auc),
#         lw=2, alpha=.8)

# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic Curve')
# plt.legend(loc="lower right")
# plt.savefig('dmfldaroc01.png')
# plt.show()

#           #pr
# plt.figure()
# plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
#         label='Chance', alpha=.8)
# lab = 'Overall AUC=%.4f' % (mean_aupr)
# print(lab)
# plt.plot(mean_recall, mean_precision, label=r'Mean PR (AUPR = %0.4f)' % (mean_aupr),
#         lw=2, alpha=.8)
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.title('Precision/Recall Curve')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.legend(loc="upper right")
# plt.savefig('dmfldapr01.png')
# plt.show()
