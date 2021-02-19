# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 16:20:05 2021

@author: chenmeijun
"""


from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import numpy as np


# In[]  数据集1 train2697

#读取数据
import pickle
pkl_file=open('p2697crf1_meanpr1928_0epoch220drop0.5lr0.001.pkl','rb')
GCMC_tprs =pickle.load(pkl_file)
GCMC_mean_auc =pickle.load(pkl_file)
GCMC_std_auc =pickle.load(pkl_file)
GCMC_std_tpr =pickle.load(pkl_file)
GCMC_y_real =pickle.load(pkl_file)
GCMC_y_proba =pickle.load(pkl_file)
GCMC_mean_aupr =pickle.load(pkl_file)
pkl_file.close()


import pickle
pkl_file=open('p2697NOcrf1_19282.pkl','rb')
p2697NOcrf_tprs =pickle.load(pkl_file)
p2697NOcrf_mean_auc =pickle.load(pkl_file)
p2697NOcrf_std_auc =pickle.load(pkl_file)
p2697NOcrf_std_tpr =pickle.load(pkl_file)
p2697NOcrf_y_real =pickle.load(pkl_file)
p2697NOcrf_y_proba =pickle.load(pkl_file)
p2697NOcrf_mean_aupr =pickle.load(pkl_file)
pkl_file.close()

#  ROC曲线
mean_fpr = np.linspace(0, 1, 100)
GCMC_mean_tpr = np.mean(GCMC_tprs, axis=0)
GCMC_mean_tpr[-1] = 1.0
p2697NOcrf_mean_tpr = np.mean(p2697NOcrf_tprs, axis=0)
p2697NOcrf_mean_tpr[-1] = 1.0

p1, = plt.plot(mean_fpr, GCMC_mean_tpr, 
          label=r'GCRFLDA with crf layer = %0.4f' % (GCMC_mean_auc),
          lw=2, alpha=.8)

p2, = plt.plot(mean_fpr, p2697NOcrf_mean_tpr, 
          label=r'GCRFLDA without crf layer = %0.4f' % (p2697NOcrf_mean_auc),
          lw=2, alpha=.8)

plt.legend(loc="lower right")
plt.xlabel("True Positive Rate")
plt.ylabel("False Positive Rate")
# plt.title('Receiver operating characteristic Curve', fontsize='large', fontweight='bold')
# plt.title('ROC curves and AUCs', fontsize='large', fontweight='bold')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
# plt.savefig('rocDataset1_crf01.png', dpi=600)
plt.savefig('GCRFrocDataset1_crf01.png', dpi=600)
plt.show()


#  PR曲线
GCMC_mean_precision, GCMC_mean_recall, _ = metrics.precision_recall_curve(GCMC_y_real, GCMC_y_proba)
p2697NOcrf_mean_precision, p2697NOcrf_mean_recall, _ = metrics.precision_recall_curve(p2697NOcrf_y_real, p2697NOcrf_y_proba)

p1, = plt.plot(GCMC_mean_recall, GCMC_mean_precision, label=r'GCRFLDA with crf layer = %0.4f' % (GCMC_mean_aupr),
         lw=2, color='r', alpha=.8)

p2, = plt.plot(p2697NOcrf_mean_recall, p2697NOcrf_mean_precision, label=r'GCRFLDA without crf layer = %0.4f' % (p2697NOcrf_mean_aupr),
         lw=2, alpha=.8)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
# plt.title('PR Curves and AUPRs')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="upper right")
plt.savefig('GCRFprDataset1_crf01.png', dpi=600)
plt.show()



import pickle
pkl_file=open('p2697ONE_cos1.pkl','rb')
p2697ONE_cos1_tprs =pickle.load(pkl_file)
p2697ONE_cos1_mean_auc =pickle.load(pkl_file)
p2697ONE_cos1_std_auc =pickle.load(pkl_file)
p2697ONE_cos1_std_tpr =pickle.load(pkl_file)
p2697ONE_cos1_y_real =pickle.load(pkl_file)
p2697ONE_cos1_y_proba =pickle.load(pkl_file)
p2697ONE_cos1_mean_aupr =pickle.load(pkl_file)
pkl_file.close()


import pickle
pkl_file=open('p2697ONE_gssim1.pkl','rb')
p2697ONE_gssim1_tprs =pickle.load(pkl_file)
p2697ONE_gssim1_mean_auc =pickle.load(pkl_file)
p2697ONE_gssim1_std_auc =pickle.load(pkl_file)
p2697ONE_gssim1_std_tpr =pickle.load(pkl_file)
p2697ONE_gssim1_y_real =pickle.load(pkl_file)
p2697ONE_gssim1_y_proba =pickle.load(pkl_file)
p2697ONE_gssim1_mean_aupr =pickle.load(pkl_file)
pkl_file.close()


#  ROC曲线
mean_fpr = np.linspace(0, 1, 100)
GCMC_mean_tpr = np.mean(GCMC_tprs, axis=0)
GCMC_mean_tpr[-1] = 1.0
p2697ONE_cos1_mean_tpr = np.mean(p2697ONE_cos1_tprs, axis=0)
p2697ONE_cos1_mean_tpr[-1] = 1.0
p2697ONE_gssim1_mean_tpr = np.mean(p2697ONE_gssim1_tprs, axis=0)
p2697ONE_gssim1_mean_tpr[-1] = 1.0

p1, = plt.plot(mean_fpr, GCMC_mean_tpr, 
          label=r'GCRFLDA with two similarity = %0.4f' % (GCMC_mean_auc),
          lw=2, alpha=.8)

p2, = plt.plot(mean_fpr, p2697ONE_cos1_mean_tpr, 
          label=r'GCRFLDA with cosine = %0.4f' % (p2697ONE_cos1_mean_auc),
          lw=2, alpha=.8)

p3, = plt.plot(mean_fpr, p2697ONE_gssim1_mean_tpr, 
          label=r'GCRFLDA with gaussian = %0.4f' % (p2697ONE_gssim1_mean_auc),
          lw=2, alpha=.8)

plt.legend(loc="lower right")
plt.xlabel("True Positive Rate")
plt.ylabel("False Positive Rate")
# plt.title('Receiver operating characteristic Curve', fontsize='large', fontweight='bold')
# plt.title('ROC curves and AUCs', fontsize='large', fontweight='bold')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.savefig('GCRFrocDataset1_ONE01.png', dpi=600)
plt.show()


#  PR曲线
GCMC_mean_precision, GCMC_mean_recall, _ = metrics.precision_recall_curve(GCMC_y_real, GCMC_y_proba)
p2697ONE_cos1_mean_precision, p2697ONE_cos1_mean_recall, _ = metrics.precision_recall_curve(p2697ONE_cos1_y_real, p2697ONE_cos1_y_proba)
p2697ONE_gssim1_mean_precision, p2697ONE_gssim1_mean_recall, _ = metrics.precision_recall_curve(p2697ONE_gssim1_y_real, p2697ONE_gssim1_y_proba)

p1, = plt.plot(GCMC_mean_recall, GCMC_mean_precision, label=r'GCRFLDA with two similarity = %0.4f' % (GCMC_mean_aupr),
         lw=2, color='r', alpha=.8)

p2, = plt.plot(p2697ONE_cos1_mean_recall, p2697ONE_cos1_mean_precision, label=r'GCRFLDA with cosine = %0.4f' % (p2697ONE_cos1_mean_aupr),
         lw=2, alpha=.8)

p3, = plt.plot(p2697ONE_gssim1_mean_recall, p2697ONE_gssim1_mean_precision, label=r'GCRFLDA with gaussian = %0.4f' % (p2697ONE_gssim1_mean_aupr),
         lw=2, alpha=.8)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
# plt.title('PR Curves and AUPRs')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="upper right")
plt.savefig('GCRFprDataset1_ONE01.png', dpi=600)
plt.show()


# In[]   数据集2 DMFLDA

from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import numpy as np

 #读取数据
import pickle
pkl_file=open('DMFLDA0epoch220drop0.75lr0.001.pkl','rb')
DMFLDA_tprs =pickle.load(pkl_file)
DMFLDA_mean_auc =pickle.load(pkl_file)
DMFLDA_std_auc =pickle.load(pkl_file)
DMFLDA_std_tpr =pickle.load(pkl_file)
DMFLDA_y_real =pickle.load(pkl_file)
DMFLDA_y_proba =pickle.load(pkl_file)
DMFLDA_mean_aupr =pickle.load(pkl_file)
pkl_file.close()


import pickle
pkl_file=open('DMFLDA_NOcrf_19190.pkl','rb')
DMFLDA_NOcrf_19190_tprs =pickle.load(pkl_file)
DMFLDA_NOcrf_19190_mean_auc =pickle.load(pkl_file)
DMFLDA_NOcrf_19190_std_auc =pickle.load(pkl_file)
DMFLDA_NOcrf_19190_std_tpr =pickle.load(pkl_file)
DMFLDA_NOcrf_19190_y_real =pickle.load(pkl_file)
DMFLDA_NOcrf_19190_y_proba =pickle.load(pkl_file)
DMFLDA_NOcrf_19190_mean_aupr =pickle.load(pkl_file)
pkl_file.close()

#  ROC曲线
mean_fpr = np.linspace(0, 1, 100)
DMFLDA_mean_tpr = np.mean(DMFLDA_tprs, axis=0)
DMFLDA_mean_tpr[-1] = 1.0
DMFLDA_NOcrf_19190_mean_tpr = np.mean(DMFLDA_NOcrf_19190_tprs, axis=0)
DMFLDA_NOcrf_19190_mean_tpr[-1] = 1.0

# plt.figure(figsize=(12,8))
p1, = plt.plot(mean_fpr, DMFLDA_mean_tpr, 
          label=r'GCRFLDA with crf layer = %0.4f' % (DMFLDA_mean_auc),
          lw=2, alpha=.8)

p2, = plt.plot(mean_fpr, DMFLDA_NOcrf_19190_mean_tpr, 
          label=r'GCRFLDA without crf layer = %0.4f' % (DMFLDA_NOcrf_19190_mean_auc),
          lw=2, alpha=.8)

plt.legend(loc="lower right")
plt.xlabel("True Positive Rate")
plt.ylabel("False Positive Rate")
# plt.title('Receiver operating characteristic Curve', fontsize='large', fontweight='bold')
# plt.title('ROC curves and AUCs', fontsize='large', fontweight='bold')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.savefig('GCRFrocDataset2_crf01.png', dpi=600)
plt.show()


#  PR曲线
DMFLDA_mean_precision, DMFLDA_mean_recall, _ = metrics.precision_recall_curve(DMFLDA_y_real, DMFLDA_y_proba)
DMFLDA_NOcrf_19190_mean_precision, DMFLDA_NOcrf_19190_mean_recall, _ = metrics.precision_recall_curve(DMFLDA_NOcrf_19190_y_real, DMFLDA_NOcrf_19190_y_proba)

p1, = plt.plot(DMFLDA_mean_recall, DMFLDA_mean_precision, label=r'GCRFLDA with crf layer = %0.4f' % (DMFLDA_mean_aupr),
         lw=2, color='r', alpha=.8)

p2, = plt.plot(DMFLDA_NOcrf_19190_mean_recall, DMFLDA_NOcrf_19190_mean_precision, label=r'GCRFLDA without crf layer = %0.4f' % (DMFLDA_NOcrf_19190_mean_aupr),
         lw=2, alpha=.8)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
# plt.title('PR Curves and AUPRs')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="upper right")
plt.savefig('GCRFprDataset2_crf01.png', dpi=600)
plt.show()


import pickle
pkl_file=open('DMFLDA_ONE_cos1.pkl','rb')
DMFLDA_ONE_cos1_tprs =pickle.load(pkl_file)
DMFLDA_ONE_cos1_mean_auc =pickle.load(pkl_file)
DMFLDA_ONE_cos1_std_auc =pickle.load(pkl_file)
DMFLDA_ONE_cos1_std_tpr =pickle.load(pkl_file)
DMFLDA_ONE_cos1_y_real =pickle.load(pkl_file)
DMFLDA_ONE_cos1_y_proba =pickle.load(pkl_file)
DMFLDA_ONE_cos1_mean_aupr =pickle.load(pkl_file)
pkl_file.close()


import pickle
pkl_file=open('DMFLDA_ONE_gssim0.pkl','rb')
DMFLDA_ONE_gssim0_tprs =pickle.load(pkl_file)
DMFLDA_ONE_gssim0_mean_auc =pickle.load(pkl_file)
DMFLDA_ONE_gssim0_std_auc =pickle.load(pkl_file)
DMFLDA_ONE_gssim0_std_tpr =pickle.load(pkl_file)
DMFLDA_ONE_gssim0_y_real =pickle.load(pkl_file)
DMFLDA_ONE_gssim0_y_proba =pickle.load(pkl_file)
DMFLDA_ONE_gssim0_mean_aupr =pickle.load(pkl_file)
pkl_file.close()


#  ROC曲线
mean_fpr = np.linspace(0, 1, 100)
DMFLDA_mean_tpr = np.mean(DMFLDA_tprs, axis=0)
DMFLDA_mean_tpr[-1] = 1.0
DMFLDA_ONE_cos1_mean_tpr = np.mean(DMFLDA_ONE_cos1_tprs, axis=0)
DMFLDA_ONE_cos1_mean_tpr[-1] = 1.0
DMFLDA_ONE_gssim0_mean_tpr = np.mean(DMFLDA_ONE_gssim0_tprs, axis=0)
DMFLDA_ONE_gssim0_mean_tpr[-1] = 1.0

p1, = plt.plot(mean_fpr, DMFLDA_mean_tpr, 
          label=r'GCRFLDA with two similarity = %0.4f' % (DMFLDA_mean_auc),
          lw=2, alpha=.8)

p2, = plt.plot(mean_fpr, DMFLDA_ONE_cos1_mean_tpr, 
          label=r'GCRFLDA with cosine = %0.4f' % (DMFLDA_ONE_cos1_mean_auc),
          lw=2, alpha=.8)

p3, = plt.plot(mean_fpr, DMFLDA_ONE_gssim0_mean_tpr, 
          label=r'GCRFLDA with gaussian = %0.4f' % (DMFLDA_ONE_gssim0_mean_auc),
          lw=2, alpha=.8)

plt.legend(loc="lower right")
plt.xlabel("True Positive Rate")
plt.ylabel("False Positive Rate")
# plt.title('Receiver operating characteristic Curve', fontsize='large', fontweight='bold')
# plt.title('ROC curves and AUCs', fontsize='large', fontweight='bold')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.savefig('GCRFrocDataset2_ONE01.png', dpi=600)
plt.show()


#  PR曲线
DMFLDA_mean_precision, DMFLDA_mean_recall, _ = metrics.precision_recall_curve(DMFLDA_y_real, DMFLDA_y_proba)
DMFLDA_ONE_cos1_mean_precision, DMFLDA_ONE_cos1_mean_recall, _ = metrics.precision_recall_curve(DMFLDA_ONE_cos1_y_real, DMFLDA_ONE_cos1_y_proba)
DMFLDA_ONE_gssim0_mean_precision, DMFLDA_ONE_gssim0_mean_recall, _ = metrics.precision_recall_curve(DMFLDA_ONE_gssim0_y_real, DMFLDA_ONE_gssim0_y_proba)

p1, = plt.plot(DMFLDA_mean_recall, DMFLDA_mean_precision, label=r'GCRFLDA with two similarity = %0.4f' % (DMFLDA_mean_aupr),
         lw=2, color='r', alpha=.8)

p2, = plt.plot(DMFLDA_ONE_cos1_mean_recall, DMFLDA_ONE_cos1_mean_precision, label=r'GCRFLDA with cosine = %0.4f' % (DMFLDA_ONE_cos1_mean_aupr),
         lw=2, alpha=.8)

p3, = plt.plot(DMFLDA_ONE_gssim0_mean_recall, DMFLDA_ONE_gssim0_mean_precision, label=r'GCRFLDA with gaussian = %0.4f' % (DMFLDA_ONE_gssim0_mean_aupr),
         lw=2, alpha=.8)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
# plt.title('PR Curves and AUPRs')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="upper right")
plt.savefig('GCRFprDataset2_ONE01.png', dpi=600)
plt.show()


# In[]

from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import numpy as np

 #读取数据  
# import pickle
# pkl_file=open('LNSUBRW0epoch180drop0.5lr0.001.pkl','rb')
# LNSUBRW_tprs =pickle.load(pkl_file)
# LNSUBRW_mean_auc =pickle.load(pkl_file)
# LNSUBRW_std_auc =pickle.load(pkl_file)
# LNSUBRW_std_tpr =pickle.load(pkl_file)
# LNSUBRW_y_real =pickle.load(pkl_file)
# LNSUBRW_y_proba =pickle.load(pkl_file)
# LNSUBRW_mean_aupr =pickle.load(pkl_file)
# pkl_file.close()

import pickle
pkl_file=open('LNSUBRW1epoch180drop0.75lr0.001.pkl','rb')
LNSUBRW_tprs =pickle.load(pkl_file)
LNSUBRW_mean_auc =pickle.load(pkl_file)
LNSUBRW_std_auc =pickle.load(pkl_file)
LNSUBRW_std_tpr =pickle.load(pkl_file)
LNSUBRW_y_real =pickle.load(pkl_file)
LNSUBRW_y_proba =pickle.load(pkl_file)
LNSUBRW_mean_aupr =pickle.load(pkl_file)
pkl_file.close()


import pickle
pkl_file=open('LNSUBRW_NOcrf1_19372.pkl','rb')
LNSUBRW_NOcrf1_19372_tprs =pickle.load(pkl_file)
LNSUBRW_NOcrf1_19372_mean_auc =pickle.load(pkl_file)
LNSUBRW_NOcrf1_19372_std_auc =pickle.load(pkl_file)
LNSUBRW_NOcrf1_19372_std_tpr =pickle.load(pkl_file)
LNSUBRW_NOcrf1_19372_y_real =pickle.load(pkl_file)
LNSUBRW_NOcrf1_19372_y_proba =pickle.load(pkl_file)
LNSUBRW_NOcrf1_19372_mean_aupr =pickle.load(pkl_file)
pkl_file.close()


#  ROC曲线
mean_fpr = np.linspace(0, 1, 100)
LNSUBRW_mean_tpr = np.mean(LNSUBRW_tprs, axis=0)
LNSUBRW_mean_tpr[-1] = 1.0
LNSUBRW_NOcrf1_19372_mean_tpr = np.mean(LNSUBRW_NOcrf1_19372_tprs, axis=0)
LNSUBRW_NOcrf1_19372_mean_tpr[-1] = 1.0

p1, = plt.plot(mean_fpr, LNSUBRW_mean_tpr, 
          label=r'GCRFLDA with crf layer = %0.4f' % (LNSUBRW_mean_auc),
          lw=2, alpha=.8)

p2, = plt.plot(mean_fpr, LNSUBRW_NOcrf1_19372_mean_tpr, 
          label=r'GCRFLDA without crf layer = %0.4f' % (LNSUBRW_NOcrf1_19372_mean_auc),
          lw=2, alpha=.8)

plt.legend(loc="lower right")
plt.xlabel("True Positive Rate")
plt.ylabel("False Positive Rate")
# plt.title('Receiver operating characteristic Curve', fontsize='large', fontweight='bold')
# plt.title('ROC curves and AUCs', fontsize='large', fontweight='bold')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.savefig('GCRFrocDataset3_crf01.png', dpi=600)
plt.show()


#  PR曲线
LNSUBRW_mean_precision, LNSUBRW_mean_recall, _ = metrics.precision_recall_curve(LNSUBRW_y_real, LNSUBRW_y_proba)
LNSUBRW_NOcrf1_19372_mean_precision, LNSUBRW_NOcrf1_19372_mean_recall, _ = metrics.precision_recall_curve(LNSUBRW_NOcrf1_19372_y_real, LNSUBRW_NOcrf1_19372_y_proba)

p1, = plt.plot(LNSUBRW_mean_recall, LNSUBRW_mean_precision, label=r'GCRFLDA with crf layer = %0.4f' % (LNSUBRW_mean_aupr),
         lw=2, color='r', alpha=.8)

p2, = plt.plot(LNSUBRW_NOcrf1_19372_mean_recall, LNSUBRW_NOcrf1_19372_mean_precision, label=r'GCRFLDA without crf layer = %0.4f' % (LNSUBRW_NOcrf1_19372_mean_aupr),
         lw=2, alpha=.8)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
# plt.title('PR Curves and AUPRs')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="upper right")
plt.savefig('GCRFprDataset3_crf01.png', dpi=600)
plt.show()


import pickle
pkl_file=open('LNSUBRWONE_cos_1937_1.pkl','rb')
LNSUBRWONE_cos_1937_1_tprs =pickle.load(pkl_file)
LNSUBRWONE_cos_1937_1_mean_auc =pickle.load(pkl_file)
LNSUBRWONE_cos_1937_1_std_auc =pickle.load(pkl_file)
LNSUBRWONE_cos_1937_1_std_tpr =pickle.load(pkl_file)
LNSUBRWONE_cos_1937_1_y_real =pickle.load(pkl_file)
LNSUBRWONE_cos_1937_1_y_proba =pickle.load(pkl_file)
LNSUBRWONE_cos_1937_1_mean_aupr =pickle.load(pkl_file)
pkl_file.close()


import pickle
pkl_file=open('LNSUBRWONE_gssim_1937_0.pkl','rb')
LNSUBRWONE_gssim_1937_0_tprs =pickle.load(pkl_file)
LNSUBRWONE_gssim_1937_0_mean_auc =pickle.load(pkl_file)
LNSUBRWONE_gssim_1937_0_std_auc =pickle.load(pkl_file)
LNSUBRWONE_gssim_1937_0_std_tpr =pickle.load(pkl_file)
LNSUBRWONE_gssim_1937_0_y_real =pickle.load(pkl_file)
LNSUBRWONE_gssim_1937_0_y_proba =pickle.load(pkl_file)
LNSUBRWONE_gssim_1937_0_mean_aupr =pickle.load(pkl_file)
pkl_file.close()


#  ROC曲线
mean_fpr = np.linspace(0, 1, 100)
LNSUBRW_mean_tpr = np.mean(LNSUBRW_tprs, axis=0)
LNSUBRW_mean_tpr[-1] = 1.0
LNSUBRWONE_cos_1937_1_mean_tpr = np.mean(LNSUBRWONE_cos_1937_1_tprs, axis=0)
LNSUBRWONE_cos_1937_1_mean_tpr[-1] = 1.0
LNSUBRWONE_gssim_1937_0_mean_tpr = np.mean(LNSUBRWONE_gssim_1937_0_tprs, axis=0)
LNSUBRWONE_gssim_1937_0_mean_tpr[-1] = 1.0

p1, = plt.plot(mean_fpr, LNSUBRW_mean_tpr, 
          label=r'GCRFLDA with two similarity = %0.4f' % (LNSUBRW_mean_auc),
          lw=2, alpha=.8)

p2, = plt.plot(mean_fpr, LNSUBRWONE_cos_1937_1_mean_tpr, 
          label=r'GCRFLDA with cosine = %0.4f' % (LNSUBRWONE_cos_1937_1_mean_auc),
          lw=2, alpha=.8)

p3, = plt.plot(mean_fpr, LNSUBRWONE_gssim_1937_0_mean_tpr, 
          label=r'GCRFLDA with gaussian = %0.4f' % (LNSUBRWONE_gssim_1937_0_mean_auc),
          lw=2, alpha=.8)

plt.legend(loc="lower right")
plt.xlabel("True Positive Rate")
plt.ylabel("False Positive Rate")
# plt.title('Receiver operating characteristic Curve', fontsize='large', fontweight='bold')
# plt.title('ROC curves and AUCs', fontsize='large', fontweight='bold')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.savefig('GCRFrocDataset3_ONE01.png', dpi=600)
plt.show()


#  PR曲线
LNSUBRW_mean_precision, LNSUBRW_mean_recall, _ = metrics.precision_recall_curve(LNSUBRW_y_real, LNSUBRW_y_proba)
LNSUBRWONE_cos_1937_1_mean_precision, LNSUBRWONE_cos_1937_1_mean_recall, _ = metrics.precision_recall_curve(LNSUBRWONE_cos_1937_1_y_real, LNSUBRWONE_cos_1937_1_y_proba)
LNSUBRWONE_gssim_1937_0_mean_precision, LNSUBRWONE_gssim_1937_0_mean_recall, _ = metrics.precision_recall_curve(LNSUBRWONE_gssim_1937_0_y_real, LNSUBRWONE_gssim_1937_0_y_proba)

p1, = plt.plot(LNSUBRW_mean_recall, LNSUBRW_mean_precision, label=r'GCRFLDA with two similarity = %0.4f' % (LNSUBRW_mean_aupr),
         lw=2, color='r', alpha=.8)

p2, = plt.plot(LNSUBRWONE_cos_1937_1_mean_recall, LNSUBRWONE_cos_1937_1_mean_precision, label=r'GCRFLDA with cosine = %0.4f' % (LNSUBRWONE_cos_1937_1_mean_aupr),
         lw=2, alpha=.8)

p3, = plt.plot(LNSUBRWONE_gssim_1937_0_mean_recall, LNSUBRWONE_gssim_1937_0_mean_precision, label=r'GCRFLDA with gaussian = %0.4f' % (LNSUBRWONE_gssim_1937_0_mean_aupr),
         lw=2, alpha=.8)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
# plt.title('PR Curves and AUPRs')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="upper right")
plt.savefig('GCRFprDataset2_ONE01.png', dpi=600)
plt.show()


 # In[]   数据集4 LDNFSGB

# from sklearn import metrics
# import matplotlib.pyplot as plt
# from sklearn.metrics import auc
# import numpy as np
 
#  #读取数据
# import pickle
# pkl_file=open('LDNFSGB10cv_19epoch280drop0.5lr0.01.pkl','rb')
# LDNFSGB_tprs =pickle.load(pkl_file)
# LDNFSGB_mean_auc =pickle.load(pkl_file)
# LDNFSGB_std_auc =pickle.load(pkl_file)
# LDNFSGB_std_tpr =pickle.load(pkl_file)
# LDNFSGB_y_real =pickle.load(pkl_file)
# LDNFSGB_y_proba =pickle.load(pkl_file)
# LDNFSGB_mean_aupr =pickle.load(pkl_file)
# pkl_file.close()

# import pickle
# pkl_file=open('LDNFSGB10cv3530_NOcrf1_19191.pkl','rb')
# LDNFSGB10cv3530_NOcrf1_19191_tprs =pickle.load(pkl_file)
# LDNFSGB10cv3530_NOcrf1_19191_mean_auc =pickle.load(pkl_file)
# LDNFSGB10cv3530_NOcrf1_19191_std_auc =pickle.load(pkl_file)
# LDNFSGB10cv3530_NOcrf1_19191_std_tpr =pickle.load(pkl_file)
# LDNFSGB10cv3530_NOcrf1_19191_y_real =pickle.load(pkl_file)
# LDNFSGB10cv3530_NOcrf1_19191_y_proba =pickle.load(pkl_file)
# LDNFSGB10cv3530_NOcrf1_19191_mean_aupr =pickle.load(pkl_file)
# pkl_file.close()

# #  ROC曲线
# mean_fpr = np.linspace(0, 1, 100)
# LDNFSGB_mean_tpr = np.mean(LDNFSGB_tprs, axis=0)
# LDNFSGB_mean_tpr[-1] = 1.0
# LDNFSGB10cv3530_NOcrf1_19191_mean_tpr = np.mean(LDNFSGB10cv3530_NOcrf1_19191_tprs, axis=0)
# LDNFSGB10cv3530_NOcrf1_19191_mean_tpr[-1] = 1.0

# p1, = plt.plot(mean_fpr, LDNFSGB_mean_tpr, 
#           label=r'GCMCLDA with crf layer = %0.4f' % (LDNFSGB_mean_auc),
#           lw=2, alpha=.8)

# p2, = plt.plot(mean_fpr, LDNFSGB10cv3530_NOcrf1_19191_mean_tpr, 
#           label=r'GCMCLDA without crf layer = %0.4f' % (LDNFSGB10cv3530_NOcrf1_19191_mean_auc),
#           lw=2, alpha=.8)

# plt.legend(loc="lower right")
# plt.xlabel("True Positive Rate")
# plt.ylabel("False Positive Rate")
# # plt.title('Receiver operating characteristic Curve', fontsize='large', fontweight='bold')
# # plt.title('ROC curves and AUCs', fontsize='large', fontweight='bold')
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.savefig('rocDataset4_crf01.png', dpi=600)
# plt.show()


# #  PR曲线
# LDNFSGB_mean_precision, LDNFSGB_mean_recall, _ = metrics.precision_recall_curve(LDNFSGB_y_real, LDNFSGB_y_proba)
# LDNFSGB10cv3530_NOcrf1_19191_mean_precision, LDNFSGB10cv3530_NOcrf1_19191_mean_recall, _ = metrics.precision_recall_curve(LDNFSGB10cv3530_NOcrf1_19191_y_real, LDNFSGB10cv3530_NOcrf1_19191_y_proba)

# p1, = plt.plot(LDNFSGB_mean_recall, LDNFSGB_mean_precision, label=r'GCMCLDA with crf layer = %0.4f' % (LDNFSGB_mean_aupr),
#          lw=2, color='r', alpha=.8)

# p2, = plt.plot(LDNFSGB10cv3530_NOcrf1_19191_mean_recall, LDNFSGB10cv3530_NOcrf1_19191_mean_precision, label=r'GCMCLDA without crf layer = %0.4f' % (LDNFSGB10cv3530_NOcrf1_19191_mean_aupr),
#          lw=2, alpha=.8)

# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# # plt.title('PR Curves and AUPRs')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.legend(loc="upper right")
# plt.savefig('prDataset4_crf01.png', dpi=600)
# plt.show()


# import pickle
# pkl_file=open('LDNFSGB10CV3530_ONE__cos2.pkl','rb')
# LDNFSGB10CV3530_ONE__cos2_tprs =pickle.load(pkl_file)
# LDNFSGB10CV3530_ONE__cos2_mean_auc =pickle.load(pkl_file)
# LDNFSGB10CV3530_ONE__cos2_std_auc =pickle.load(pkl_file)
# LDNFSGB10CV3530_ONE__cos2_std_tpr =pickle.load(pkl_file)
# LDNFSGB10CV3530_ONE__cos2_y_real =pickle.load(pkl_file)
# LDNFSGB10CV3530_ONE__cos2_y_proba =pickle.load(pkl_file)
# LDNFSGB10CV3530_ONE__cos2_mean_aupr =pickle.load(pkl_file)
# pkl_file.close()

# import pickle
# pkl_file=open('LDNFSGB10CV3530_ONE_gssim0.pkl','rb')
# LDNFSGB10CV3530_ONE_gssim0_tprs =pickle.load(pkl_file)
# LDNFSGB10CV3530_ONE_gssim0_mean_auc =pickle.load(pkl_file)
# LDNFSGB10CV3530_ONE_gssim0_std_auc =pickle.load(pkl_file)
# LDNFSGB10CV3530_ONE_gssim0_std_tpr =pickle.load(pkl_file)
# LDNFSGB10CV3530_ONE_gssim0_y_real =pickle.load(pkl_file)
# LDNFSGB10CV3530_ONE_gssim0_y_proba =pickle.load(pkl_file)
# LDNFSGB10CV3530_ONE_gssim0_mean_aupr =pickle.load(pkl_file)
# pkl_file.close()


# #  ROC曲线
# mean_fpr = np.linspace(0, 1, 100)
# LDNFSGB_mean_tpr = np.mean(LDNFSGB_tprs, axis=0)
# LDNFSGB_mean_tpr[-1] = 1.0
# LDNFSGB10CV3530_ONE__cos2_mean_tpr = np.mean(LDNFSGB10CV3530_ONE__cos2_tprs, axis=0)
# LDNFSGB10CV3530_ONE__cos2_mean_tpr[-1] = 1.0
# LDNFSGB10CV3530_ONE_gssim0_mean_tpr = np.mean(LDNFSGB10CV3530_ONE_gssim0_tprs, axis=0)
# LDNFSGB10CV3530_ONE_gssim0_mean_tpr[-1] = 1.0

# p1, = plt.plot(mean_fpr, LDNFSGB_mean_tpr, 
#           label=r'GCMCLDA with two similarity = %0.4f' % (LDNFSGB_mean_auc),
#           lw=2, alpha=.8)

# p2, = plt.plot(mean_fpr, LDNFSGB10CV3530_ONE__cos2_mean_tpr, 
#           label=r'GCMCLDA with cosine = %0.4f' % (LDNFSGB10CV3530_ONE__cos2_mean_auc),
#           lw=2, alpha=.8)

# p3, = plt.plot(mean_fpr, LDNFSGB10CV3530_ONE_gssim0_mean_tpr, 
#           label=r'GCMCLDA with gaussian = %0.4f' % (LDNFSGB10CV3530_ONE_gssim0_mean_auc),
#           lw=2, alpha=.8)

# plt.legend(loc="lower right")
# plt.xlabel("True Positive Rate")
# plt.ylabel("False Positive Rate")
# # plt.title('Receiver operating characteristic Curve', fontsize='large', fontweight='bold')
# # plt.title('ROC curves and AUCs', fontsize='large', fontweight='bold')
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.savefig('rocDataset4_ONE01.png', dpi=600)
# plt.show()


# #  PR曲线
# LDNFSGB_mean_precision, LDNFSGB_mean_recall, _ = metrics.precision_recall_curve(LDNFSGB_y_real, LDNFSGB_y_proba)
# LDNFSGB10CV3530_ONE__cos2_mean_precision, LDNFSGB10CV3530_ONE__cos2_mean_recall, _ = metrics.precision_recall_curve(LDNFSGB10CV3530_ONE__cos2_y_real, LDNFSGB10CV3530_ONE__cos2_y_proba)
# LDNFSGB10CV3530_ONE_gssim0_mean_precision, LDNFSGB10CV3530_ONE_gssim0_mean_recall, _ = metrics.precision_recall_curve(LDNFSGB10CV3530_ONE_gssim0_y_real, LDNFSGB10CV3530_ONE_gssim0_y_proba)

# p1, = plt.plot(LDNFSGB_mean_recall, LDNFSGB_mean_precision, label=r'GCMCLDA with two similarity = %0.4f' % (LDNFSGB_mean_aupr),
#          lw=2, color='r', alpha=.8)

# p2, = plt.plot(LDNFSGB10CV3530_ONE__cos2_mean_recall, LDNFSGB10CV3530_ONE__cos2_mean_precision, label=r'GCMCLDA with cosine = %0.4f' % (LDNFSGB10CV3530_ONE__cos2_mean_aupr),
#          lw=2, alpha=.8)

# p3, = plt.plot(LDNFSGB10CV3530_ONE_gssim0_mean_recall, LDNFSGB10CV3530_ONE_gssim0_mean_precision, label=r'GCMCLDA with gaussian = %0.4f' % (LDNFSGB10CV3530_ONE_gssim0_mean_aupr),
#          lw=2, alpha=.8)

# plt.legend(loc="lower right")
# plt.xlabel("True Positive Rate")
# plt.ylabel("False Positive Rate")
# # plt.title('Receiver operating characteristic Curve', fontsize='large', fontweight='bold')
# # plt.title('ROC curves and AUCs', fontsize='large', fontweight='bold')
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.savefig('prDataset4_ONE01.png', dpi=600)
# plt.show()

# In[]   数据集4 LDNFSGB

from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import numpy as np
 
 #读取数据
import pickle
pkl_file=open('LDNFSGB_case10cv4655epoch200drop0.5lr0.001.pkl','rb')
LDNFSGB_tprs =pickle.load(pkl_file)
LDNFSGB_mean_auc =pickle.load(pkl_file)
LDNFSGB_std_auc =pickle.load(pkl_file)
LDNFSGB_std_tpr =pickle.load(pkl_file)
LDNFSGB_y_real =pickle.load(pkl_file)
LDNFSGB_y_proba =pickle.load(pkl_file)
LDNFSGB_mean_aupr =pickle.load(pkl_file)
pkl_file.close()

import pickle
pkl_file=open('LDNFSGB_case10cv_NOcrf_46550.pkl','rb')
LDNFSGB_case10cv_NOcrf_4655_0_tprs =pickle.load(pkl_file)
LDNFSGB_case10cv_NOcrf_4655_0_mean_auc =pickle.load(pkl_file)
LDNFSGB_case10cv_NOcrf_4655_0_std_auc =pickle.load(pkl_file)
LDNFSGB_case10cv_NOcrf_4655_0_std_tpr =pickle.load(pkl_file)
LDNFSGB_case10cv_NOcrf_4655_0_y_real =pickle.load(pkl_file)
LDNFSGB_case10cv_NOcrf_4655_0_y_proba =pickle.load(pkl_file)
LDNFSGB_case10cv_NOcrf_4655_0_mean_aupr =pickle.load(pkl_file)
pkl_file.close()

#  ROC曲线
mean_fpr = np.linspace(0, 1, 100)
LDNFSGB_mean_tpr = np.mean(LDNFSGB_tprs, axis=0)
LDNFSGB_mean_tpr[-1] = 1.0
LDNFSGB_case10cv_NOcrf_4655_0_mean_tpr = np.mean(LDNFSGB_case10cv_NOcrf_4655_0_tprs, axis=0)
LDNFSGB_case10cv_NOcrf_4655_0_mean_tpr[-1] = 1.0

p1, = plt.plot(mean_fpr, LDNFSGB_mean_tpr, 
          label=r'GCRFLDA with crf layer = %0.4f' % (LDNFSGB_mean_auc),
          lw=2, alpha=.8)

p2, = plt.plot(mean_fpr, LDNFSGB_case10cv_NOcrf_4655_0_mean_tpr, 
          label=r'GCRFLDA without crf layer = %0.4f' % (LDNFSGB_case10cv_NOcrf_4655_0_mean_auc),
          lw=2, alpha=.8)

plt.legend(loc="lower right")
plt.xlabel("True Positive Rate")
plt.ylabel("False Positive Rate")
# plt.title('Receiver operating characteristic Curve', fontsize='large', fontweight='bold')
# plt.title('ROC curves and AUCs', fontsize='large', fontweight='bold')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.savefig('GCRFrocDataset4_crf02.png', dpi=600)
plt.show()


#  PR曲线
LDNFSGB_mean_precision, LDNFSGB_mean_recall, _ = metrics.precision_recall_curve(LDNFSGB_y_real, LDNFSGB_y_proba)
LDNFSGB_case10cv_NOcrf_4655_0_mean_precision, LDNFSGB_case10cv_NOcrf_4655_0_mean_recall, _ = metrics.precision_recall_curve(LDNFSGB_case10cv_NOcrf_4655_0_y_real, LDNFSGB_case10cv_NOcrf_4655_0_y_proba)

p1, = plt.plot(LDNFSGB_mean_recall, LDNFSGB_mean_precision, label=r'GCRFLDA with crf layer = %0.4f' % (LDNFSGB_mean_aupr),
         lw=2, color='r', alpha=.8)

p2, = plt.plot(LDNFSGB_case10cv_NOcrf_4655_0_mean_recall, LDNFSGB_case10cv_NOcrf_4655_0_mean_precision, label=r'GCRFLDA without crf layer = %0.4f' % (LDNFSGB_case10cv_NOcrf_4655_0_mean_aupr),
         lw=2, alpha=.8)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
# plt.title('PR Curves and AUPRs')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="upper right")
plt.savefig('GCRFprDataset4_crf02.png', dpi=600)
plt.show()


import pickle
pkl_file=open('LDNFSGB_case10cv_ONE__cos0.pkl','rb')
LDNFSGB_case10cv_ONE__cos0_tprs =pickle.load(pkl_file)
LDNFSGB_case10cv_ONE__cos0_mean_auc =pickle.load(pkl_file)
LDNFSGB_case10cv_ONE__cos0_std_auc =pickle.load(pkl_file)
LDNFSGB_case10cv_ONE__cos0_std_tpr =pickle.load(pkl_file)
LDNFSGB_case10cv_ONE__cos0_y_real =pickle.load(pkl_file)
LDNFSGB_case10cv_ONE__cos0_y_proba =pickle.load(pkl_file)
LDNFSGB_case10cv_ONE__cos0_mean_aupr =pickle.load(pkl_file)
pkl_file.close()

import pickle
pkl_file=open('LDNFSGB_case10cv_ONE_gssim0.pkl','rb')
LDNFSGB_case10cv_ONE_gssim0_tprs =pickle.load(pkl_file)
LDNFSGB_case10cv_ONE_gssim0_mean_auc =pickle.load(pkl_file)
LDNFSGB_case10cv_ONE_gssim0_std_auc =pickle.load(pkl_file)
LDNFSGB_case10cv_ONE_gssim0_std_tpr =pickle.load(pkl_file)
LDNFSGB_case10cv_ONE_gssim0_y_real =pickle.load(pkl_file)
LDNFSGB_case10cv_ONE_gssim0_y_proba =pickle.load(pkl_file)
LDNFSGB_case10cv_ONE_gssim0_mean_aupr =pickle.load(pkl_file)
pkl_file.close()


#  ROC曲线
mean_fpr = np.linspace(0, 1, 100)
LDNFSGB_mean_tpr = np.mean(LDNFSGB_tprs, axis=0)
LDNFSGB_mean_tpr[-1] = 1.0
LDNFSGB_case10cv_ONE__cos0_mean_tpr = np.mean(LDNFSGB_case10cv_ONE__cos0_tprs, axis=0)
LDNFSGB_case10cv_ONE__cos0_mean_tpr[-1] = 1.0
LDNFSGB_case10cv_ONE_gssim0_mean_tpr = np.mean(LDNFSGB_case10cv_ONE_gssim0_tprs, axis=0)
LDNFSGB_case10cv_ONE_gssim0_mean_tpr[-1] = 1.0

p1, = plt.plot(mean_fpr, LDNFSGB_mean_tpr, 
          label=r'GCRFLDA with two similarity = %0.4f' % (LDNFSGB_mean_auc),
          lw=2, alpha=.8)

p2, = plt.plot(mean_fpr, LDNFSGB_case10cv_ONE__cos0_mean_tpr, 
          label=r'GCRFLDA with cosine = %0.4f' % (LDNFSGB_case10cv_ONE__cos0_mean_auc),
          lw=2, alpha=.8)

p3, = plt.plot(mean_fpr, LDNFSGB_case10cv_ONE_gssim0_mean_tpr, 
          label=r'GCRFLDA with gaussian = %0.4f' % (LDNFSGB_case10cv_ONE_gssim0_mean_auc),
          lw=2, alpha=.8)

plt.legend(loc="lower right")
plt.xlabel("True Positive Rate")
plt.ylabel("False Positive Rate")
# plt.title('Receiver operating characteristic Curve', fontsize='large', fontweight='bold')
# plt.title('ROC curves and AUCs', fontsize='large', fontweight='bold')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.savefig('GCRFrocDataset4_ONE02.png', dpi=600)
plt.show()


#  PR曲线
LDNFSGB_mean_precision, LDNFSGB_mean_recall, _ = metrics.precision_recall_curve(LDNFSGB_y_real, LDNFSGB_y_proba)
LDNFSGB_case10cv_ONE__cos0_mean_precision, LDNFSGB_case10cv_ONE__cos0_mean_recall, _ = metrics.precision_recall_curve(LDNFSGB_case10cv_ONE__cos0_y_real, LDNFSGB_case10cv_ONE__cos0_y_proba)
LDNFSGB_case10cv_ONE_gssim0_mean_precision, LDNFSGB_case10cv_ONE_gssim0_mean_recall, _ = metrics.precision_recall_curve(LDNFSGB_case10cv_ONE_gssim0_y_real, LDNFSGB_case10cv_ONE_gssim0_y_proba)

p1, = plt.plot(LDNFSGB_mean_recall, LDNFSGB_mean_precision, label=r'GCRFLDA with two similarity = %0.4f' % (LDNFSGB_mean_aupr),
         lw=2, color='r', alpha=.8)

p2, = plt.plot(LDNFSGB_case10cv_ONE__cos0_mean_recall, LDNFSGB_case10cv_ONE__cos0_mean_precision, label=r'GCRFLDA with cosine = %0.4f' % (LDNFSGB_case10cv_ONE__cos0_mean_aupr),
         lw=2, alpha=.8)

p3, = plt.plot(LDNFSGB_case10cv_ONE_gssim0_mean_recall, LDNFSGB_case10cv_ONE_gssim0_mean_precision, label=r'GCRFLDA with gaussian = %0.4f' % (LDNFSGB_case10cv_ONE_gssim0_mean_aupr),
         lw=2, alpha=.8)

plt.legend(loc="upper right")
plt.xlabel("True Positive Rate")
plt.ylabel("False Positive Rate")
# plt.title('Receiver operating characteristic Curve', fontsize='large', fontweight='bold')
# plt.title('ROC curves and AUCs', fontsize='large', fontweight='bold')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.savefig('GCRFprDataset4_ONE02.png', dpi=600)
plt.show()