# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 16:20:18 2021

@author: chenmeijun
"""


## 恢复默认
#import matplotlib as mpl
#mpl.rcParams.update(mpl.rcParamsDefault)

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

# 
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

#  #读取数据  LDNFSGB10cv3530
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

 #读取数据  LDNFSGB10cv所有未知关联作测试集
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


 #读取数据  fourdataset 28 37
import pickle
pkl_file=open('Fourset2837epoch200drop0.5lr0.001.pkl','rb')
Fourset_tprs =pickle.load(pkl_file)
Fourset_mean_auc =pickle.load(pkl_file)
Fourset_std_auc =pickle.load(pkl_file)
Fourset_std_tpr =pickle.load(pkl_file)
Fourset_y_real =pickle.load(pkl_file)
Fourset_y_proba =pickle.load(pkl_file)
Fourset_mean_aupr =pickle.load(pkl_file)
pkl_file.close()



from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import numpy as np
#  ROC曲线
mean_fpr = np.linspace(0, 1, 100)
GCMC_mean_tpr = np.mean(GCMC_tprs, axis=0)
GCMC_mean_tpr[-1] = 1.0
DMFLDA_mean_tpr = np.mean(DMFLDA_tprs, axis=0)
DMFLDA_mean_tpr[-1] = 1.0
LNSUBRW_mean_tpr = np.mean(LNSUBRW_tprs, axis=0)
LNSUBRW_mean_tpr[-1] = 1.0
LDNFSGB_mean_tpr = np.mean(LDNFSGB_tprs, axis=0)
LDNFSGB_mean_tpr[-1] = 1.0

Fourset_mean_tpr = np.mean(LDNFSGB_tprs, axis=0)
Fourset_mean_tpr[-1] = 1.0

#std_auc = np.std(aucs)
# p1, = plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
#          label='Chance', alpha=.8)
p1, = plt.plot(mean_fpr, GCMC_mean_tpr, 
          label=r'Dataset1 = %0.4f' % (GCMC_mean_auc),
          lw=2, alpha=.8)

p2, = plt.plot(mean_fpr, DMFLDA_mean_tpr, 
          label=r'Dataset2 = %0.4f' % (DMFLDA_mean_auc),
          lw=2, alpha=.8)

p3, = plt.plot(mean_fpr, LNSUBRW_mean_tpr, 
          label=r'Dataset3 = %0.4f' % (LNSUBRW_mean_auc),
          lw=2, alpha=.8)

p4, = plt.plot(mean_fpr, LDNFSGB_mean_tpr, 
          label=r'Dataset4 = %0.4f' % (LDNFSGB_mean_auc),
          lw=2, alpha=.8)

p5, = plt.plot(mean_fpr, Fourset_mean_tpr, 
          label=r'Whole Dataset = %0.4f' % (Fourset_mean_auc),
          lw=2, alpha=.8)

plt.legend(loc="lower right")
plt.xlabel("True Positive Rate")
plt.ylabel("False Positive Rate")
# plt.title('Receiver operating characteristic Curve', fontsize='large', fontweight='bold')
#plt.title('ROC curves and AUCs', fontsize='large', fontweight='bold')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
#plt.savefig('rocFourset_04.png', dpi=800)
plt.savefig('rocFourset_0128.png', dpi=600)
plt.show()
# plt.rcParams['savefig.dpi'] = 800 #图片像素
# plt.rcParams['figure.dpi'] = 800 #分辨率
# plt.savefig('heatmap03.jpg', dpi=800) #指定分辨率保存
# plt.show()


#  PR曲线
GCMC_mean_precision, GCMC_mean_recall, _ = metrics.precision_recall_curve(GCMC_y_real, GCMC_y_proba)
DMFLDA_mean_precision, DMFLDA_mean_recall, _ = metrics.precision_recall_curve(DMFLDA_y_real, DMFLDA_y_proba)
LNSUBRW_mean_precision, LNSUBRW_mean_recall, _ = metrics.precision_recall_curve(LNSUBRW_y_real, LNSUBRW_y_proba)
LDNFSGB_mean_precision, LDNFSGB_mean_recall, _ = metrics.precision_recall_curve(LDNFSGB_y_real, LDNFSGB_y_proba)

Fourset_mean_precision, Fourset_mean_recall, _ = metrics.precision_recall_curve(Fourset_y_real, Fourset_y_proba)

p1, = plt.plot(GCMC_mean_recall, GCMC_mean_precision, label=r'Dataset1 = %0.4f' % (GCMC_mean_aupr),
         lw=2, alpha=.8)

p2, = plt.plot(DMFLDA_mean_recall, DMFLDA_mean_precision, label=r'Dataset2 = %0.4f' % (DMFLDA_mean_aupr),
         lw=2, alpha=.8)
p3, = plt.plot(LNSUBRW_mean_recall, LNSUBRW_mean_precision, label=r'Dataset3 = %0.4f' % (LNSUBRW_mean_aupr),
         lw=2, alpha=.8)
p4, = plt.plot(LDNFSGB_mean_recall, LDNFSGB_mean_precision, label=r'Dataset4 = %0.4f' % (LDNFSGB_mean_aupr),
          lw=2, alpha=.8)

p5, = plt.plot(Fourset_mean_recall, Fourset_mean_precision, label=r'Whole Dataset = %0.4f' % (Fourset_mean_aupr),
          lw=2, alpha=.8)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
#plt.title('PR Curves and AUPRs')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="upper right")
#plt.savefig('prFourset_04.png', dpi=800)
plt.savefig('prFourset_0128.png', dpi=600)
plt.show()