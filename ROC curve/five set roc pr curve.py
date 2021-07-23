# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 14:14:05 2021

@author: chenmeijun
"""



## 恢复默认
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import numpy as np


 #Dataset 1
import pickle
pkl_file=open('Atrain2697fivefold_sim1928epoch240drop0.75lr0.0011.pkl','rb')
GCMC_aucs=pickle.load(pkl_file)
GCMC_auprcs=pickle.load(pkl_file)
GCMC_Epochnum=pickle.load(pkl_file)
GCMC_losstest_history=pickle.load(pkl_file)
GCMC_losstrian_history=pickle.load(pkl_file)
GCMC_AUCresult=pickle.load(pkl_file)
GCMC_AUPRresult=pickle.load(pkl_file)
GCMC_CI=pickle.load(pkl_file)
GCMC_tprs =pickle.load(pkl_file)
GCMC_mean_auc =pickle.load(pkl_file)
GCMC_std_auc =pickle.load(pkl_file)
GCMC_std_tpr =pickle.load(pkl_file)
GCMC_y_real =pickle.load(pkl_file)
GCMC_y_proba =pickle.load(pkl_file)
GCMC_mean_aupr =pickle.load(pkl_file)
pkl_file.close()



#Dataset 2
import pickle
pkl_file=open('DMFLDA1919epoch220drop0.75lr0.0012.pkl','rb')
DMFLDA_aucs=pickle.load(pkl_file)
DMFLDA_auprcs=pickle.load(pkl_file)
DMFLDA_Epochnum=pickle.load(pkl_file)
DMFLDA_losstest_history=pickle.load(pkl_file)
DMFLDA_losstrian_history=pickle.load(pkl_file)
DMFLDA_AUCresult=pickle.load(pkl_file)
DMFLDA_AUPRresult=pickle.load(pkl_file)
DMFLDA_CI=pickle.load(pkl_file)
DMFLDA_tprs =pickle.load(pkl_file)
DMFLDA_mean_auc =pickle.load(pkl_file)
DMFLDA_std_auc =pickle.load(pkl_file)
DMFLDA_std_tpr =pickle.load(pkl_file)
DMFLDA_y_real =pickle.load(pkl_file)
DMFLDA_y_proba =pickle.load(pkl_file)
DMFLDA_mean_aupr =pickle.load(pkl_file)
pkl_file.close()


#Dataset3
import pickle
pkl_file=open('LNSUBRW1937epoch180drop0.75lr0.0017.pkl','rb')
LNSUBRW_aucs=pickle.load(pkl_file)
LNSUBRW_auprcs=pickle.load(pkl_file)
LNSUBRW_Epochnum=pickle.load(pkl_file)
LNSUBRW_losstest_history=pickle.load(pkl_file)
LNSUBRW_losstrian_history=pickle.load(pkl_file)
LNSUBRW_AUCresult=pickle.load(pkl_file)
LNSUBRW_AUPRresult=pickle.load(pkl_file)
LNSUBRW_CI=pickle.load(pkl_file)
LNSUBRW_tprs =pickle.load(pkl_file)
LNSUBRW_mean_auc =pickle.load(pkl_file)
LNSUBRW_std_auc =pickle.load(pkl_file)
LNSUBRW_std_tpr =pickle.load(pkl_file)
LNSUBRW_y_real =pickle.load(pkl_file)
LNSUBRW_y_proba =pickle.load(pkl_file)
LNSUBRW_mean_aupr =pickle.load(pkl_file)
pkl_file.close()

#Dataset4
import pickle
pkl_file=open('LDASR7364epoch240drop0.5lr0.0018.pkl','rb')
LDNFSGB_aucs=pickle.load(pkl_file)
LDNFSGB_auprcs=pickle.load(pkl_file)
LDNFSGB_Epochnum=pickle.load(pkl_file)
LDNFSGB_losstest_history=pickle.load(pkl_file)
LDNFSGB_losstrian_history=pickle.load(pkl_file)
LDNFSGB_AUCresult=pickle.load(pkl_file)
LDNFSGB_AUPRresult=pickle.load(pkl_file)
LDNFSGB_CI=pickle.load(pkl_file)
LDNFSGB_tprs =pickle.load(pkl_file)
LDNFSGB_mean_auc =pickle.load(pkl_file)
LDNFSGB_std_auc =pickle.load(pkl_file)
LDNFSGB_std_tpr =pickle.load(pkl_file)
LDNFSGB_y_real =pickle.load(pkl_file)
LDNFSGB_y_proba =pickle.load(pkl_file)
LDNFSGB_mean_aupr =pickle.load(pkl_file)
pkl_file.close()


#wholeset
import pickle
pkl_file=open('Weight_wholeset4339_epoch240drop0.5lr0.0012837.pkl','rb')
Fourset_aucs=pickle.load(pkl_file)
Fourset_auprcs=pickle.load(pkl_file)
Fourset_Epochnum=pickle.load(pkl_file)
Fourset_losstest_history=pickle.load(pkl_file)
Fourset_losstrian_history=pickle.load(pkl_file)
Fourset_AUCresult=pickle.load(pkl_file)
Fourset_AUPRresult=pickle.load(pkl_file)
Fourset_CI=pickle.load(pkl_file)
Fourset_PKL=pickle.load(pkl_file)
Fourset_Weight=pickle.load(pkl_file)
Fourset_tprs =pickle.load(pkl_file)
Fourset_mean_auc =pickle.load(pkl_file)
Fourset_std_auc =pickle.load(pkl_file)
Fourset_std_tpr =pickle.load(pkl_file)
Fourset_y_real =pickle.load(pkl_file)
Fourset_y_proba =pickle.load(pkl_file)
Fourset_mean_aupr =pickle.load(pkl_file)
pkl_file.close()


# In[]


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
Fourset_mean_tpr = np.mean(Fourset_tprs, axis=0)
Fourset_mean_tpr[-1] = 1.0

fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
          alpha=.8)

ax.plot(mean_fpr, GCMC_mean_tpr, 
          label=r'Dataset1 (AUC = %0.4f $\pm$ %0.4f)' % (GCMC_mean_auc,GCMC_std_auc),  ###
          lw=2, alpha=.8)

 ####
GCMC_tprs_upper = np.minimum(GCMC_mean_tpr + GCMC_std_tpr, 1)   ####
GCMC_tprs_lower = np.maximum(GCMC_mean_tpr - GCMC_std_tpr, 0)   ####
ax.fill_between(mean_fpr, GCMC_tprs_lower, GCMC_tprs_upper, color='grey', alpha=.2,  ###
                 label=r'$\pm$ 1 std. dev.')

ax.plot(mean_fpr, DMFLDA_mean_tpr, 
          label=r'Dataset2 (AUC = %0.4f $\pm$ %0.4f)' % (DMFLDA_mean_auc,DMFLDA_std_auc),
          lw=2, alpha=.8)

DMFLDA_tprs_upper = np.minimum(DMFLDA_mean_tpr + DMFLDA_std_tpr, 1)   ####
DMFLDA_tprs_lower = np.maximum(DMFLDA_mean_tpr - DMFLDA_std_tpr, 0)   ####
ax.fill_between(mean_fpr, DMFLDA_tprs_lower, DMFLDA_tprs_upper, color='grey', alpha=.2)

ax.plot(mean_fpr, LNSUBRW_mean_tpr, 
          label=r'Dataset3 (AUC = %0.4f $\pm$ %0.4f)' % (LNSUBRW_mean_auc,LNSUBRW_std_auc),
          lw=2, alpha=.8)

LNSUBRW_tprs_upper = np.minimum(LNSUBRW_mean_tpr + LNSUBRW_std_tpr, 1)   ####
LNSUBRW_tprs_lower = np.maximum(LNSUBRW_mean_tpr - LNSUBRW_std_tpr, 0)   ####
ax.fill_between(mean_fpr, LNSUBRW_tprs_lower, LNSUBRW_tprs_upper, color='grey', alpha=.2)

ax.plot(mean_fpr, LDNFSGB_mean_tpr, 
          label=r'Dataset4 (AUC = %0.4f $\pm$ %0.4f)' % (LDNFSGB_mean_auc,LDNFSGB_std_auc),
          lw=2, alpha=.8)

LDNFSGB_tprs_upper = np.minimum(LDNFSGB_mean_tpr + LDNFSGB_std_tpr, 1)   ####
LDNFSGB_tprs_lower = np.maximum(LDNFSGB_mean_tpr - LDNFSGB_std_tpr, 0)   ####
ax.fill_between(mean_fpr, LDNFSGB_tprs_lower, LDNFSGB_tprs_upper, color='grey', alpha=.2)

ax.plot(mean_fpr, Fourset_mean_tpr, 
          label=r'Whole Dataset (AUC = %0.4f $\pm$ %0.4f)' % (Fourset_mean_auc,Fourset_std_auc),
          lw=2, alpha=.8)

Fourset_tprs_upper = np.minimum(Fourset_mean_tpr + Fourset_std_tpr, 1)   ####
Fourset_tprs_lower = np.maximum(Fourset_mean_tpr - Fourset_std_tpr, 0)   ####
ax.fill_between(mean_fpr, Fourset_tprs_lower, Fourset_tprs_upper, color='grey', alpha=.2)

ax.legend(loc="lower right")
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],xlabel="True Positive Rate",ylabel="False Positive Rate")

plt.savefig('FoursetROC01.png', dpi=600)
plt.show()





#  PR曲线
GCMC_mean_precision, GCMC_mean_recall, _ = metrics.precision_recall_curve(GCMC_y_real, GCMC_y_proba)
DMFLDA_mean_precision, DMFLDA_mean_recall, _ = metrics.precision_recall_curve(DMFLDA_y_real, DMFLDA_y_proba)
LNSUBRW_mean_precision, LNSUBRW_mean_recall, _ = metrics.precision_recall_curve(LNSUBRW_y_real, LNSUBRW_y_proba)
LDNFSGB_mean_precision, LDNFSGB_mean_recall, _ = metrics.precision_recall_curve(LDNFSGB_y_real, LDNFSGB_y_proba)
Fourset_mean_precision, Fourset_mean_recall, _ = metrics.precision_recall_curve(Fourset_y_real, Fourset_y_proba)

GCMC_std_aupr = np.std(GCMC_auprcs)
DMFLDA_std_aupr = np.std(DMFLDA_auprcs)
LNSUBRW_std_aupr = np.std(LNSUBRW_auprcs)
LDNFSGB_std_aupr = np.std(LDNFSGB_auprcs)
Fourset_std_aupr = np.std(Fourset_auprcs)

fig, ax = plt.subplots()
ax.plot(GCMC_mean_recall, GCMC_mean_precision, label=r'Dataset1 (AUPR = %0.4f $\pm$ %0.4f)' % (GCMC_mean_aupr, GCMC_std_aupr),
         lw=2, alpha=.8)

ax.plot(DMFLDA_mean_recall, DMFLDA_mean_precision, label=r'Dataset2 (AUPR = %0.4f $\pm$ %0.4f)' % (DMFLDA_mean_aupr, DMFLDA_std_aupr),
         lw=2, alpha=.8)

ax.plot(LNSUBRW_mean_recall, LNSUBRW_mean_precision, label=r'Dataset3 (AUPR = %0.4f $\pm$ %0.4f)' % (LNSUBRW_mean_aupr, LNSUBRW_std_aupr),
         lw=2, alpha=.8)

ax.plot(LDNFSGB_mean_recall, LDNFSGB_mean_precision, label=r'Dataset4 (AUPR = %0.4f $\pm$ %0.4f)' % (LDNFSGB_mean_aupr, LDNFSGB_std_aupr),
          lw=2, alpha=.8)

ax.plot(Fourset_mean_recall, Fourset_mean_precision, label=r'Whole Dataset (AUPR = %0.4f $\pm$ %0.4f)' % (Fourset_mean_aupr, Fourset_std_aupr),
          lw=2, alpha=.8)

ax.legend(loc="upper right")
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],xlabel="Recall",ylabel="Precision")

plt.savefig('FoursetPR01.png', dpi=600)
plt.show()


