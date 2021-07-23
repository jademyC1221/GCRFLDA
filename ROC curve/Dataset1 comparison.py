# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 14:18:43 2021

@author: chenmeijun
"""


from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import numpy as np

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

# In[] 
 #GCRFLDA
import pickle
pkl_file=open('Atrain2697fivefold_sim1928epoch240drop0.75lr0.0011.pkl','rb')
aucs=pickle.load(pkl_file)
auprcs=pickle.load(pkl_file)
Epochnum=pickle.load(pkl_file)
losstest_history=pickle.load(pkl_file)
losstrian_history=pickle.load(pkl_file)
AUCresult=pickle.load(pkl_file)
AUPRresult=pickle.load(pkl_file)
CI=pickle.load(pkl_file)
tprs =pickle.load(pkl_file)
mean_auc =pickle.load(pkl_file)
std_auc =pickle.load(pkl_file)
std_tpr =pickle.load(pkl_file)
y_real =pickle.load(pkl_file)
y_proba =pickle.load(pkl_file)
mean_aupr =pickle.load(pkl_file)
pkl_file.close()


  #DMFLDA
import pickle
pkl_file=open('DMFLDA_allneg_dataset1_0.9523.pkl','rb')
DMFLDA_tprs=pickle.load(pkl_file)
DMFLDA_aucs=pickle.load(pkl_file)
DMFLDA_ci=pickle.load(pkl_file)
DMFLDA_mean_fpr=pickle.load(pkl_file)
DMFLDA_mean_tpr=pickle.load(pkl_file)
DMFLDA_mean_auc=pickle.load(pkl_file)
DMFLDA_y_real=pickle.load(pkl_file)
DMFLDA_y_proba=pickle.load(pkl_file)
DMFLDA_mean_aupr=pickle.load(pkl_file)
DMFLDA_y_true_val=pickle.load(pkl_file)
DMFLDA_y_pred_val=pickle.load(pkl_file)
DMFLDA_tprs=pickle.load(pkl_file)
pkl_file.close()


# GAMCLDA
import pickle
pkl_file=open('GAMCLDAfivefold_0.90.pkl','rb')
GAMCLDA_FPR =pickle.load(pkl_file)
GAMCLDA_TPR =pickle.load(pkl_file)
GAMCLDA_meanauc =pickle.load(pkl_file)
GAMCLDA_meanaupr =pickle.load(pkl_file)
GAMCLDA_R =pickle.load(pkl_file)
GAMCLDA_P =pickle.load(pkl_file)
pkl_file.close()

# LDICDL
import pickle
pkl_file=open('LDICDL5fold_allneg_0.8980.pkl','rb')
LDICDL_tpr =pickle.load(pkl_file)
LDICDL_fpr =pickle.load(pkl_file)
LDICDLroc_auc =pickle.load(pkl_file)
LDICDLroc_aupr =pickle.load(pkl_file)
LDICDLmean_recall =pickle.load(pkl_file)
LDICDLmean_precision =pickle.load(pkl_file)
pkl_file.close()


#  ROC
mean_fpr = np.linspace(0, 1, 100)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
#std_auc = np.std(aucs)

fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
          alpha=.8)
ax.plot(mean_fpr, mean_tpr, color='b',
          label=r'GCRFLDA (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc,std_auc),
          lw=2, alpha=.8)

ax.plot(DMFLDA_mean_fpr, DMFLDA_mean_tpr,color='darkorange',
               label=r'DMFLDA = %0.4f' % (DMFLDA_mean_auc),
               lw=2, alpha=.8)

ax.plot(GAMCLDA_FPR, GAMCLDA_TPR,color='firebrick',
               label=r'GAMCLDA = %0.4f' % (GAMCLDA_meanauc),
               lw=2, alpha=.8)

ax.plot(LDICDL_fpr, LDICDL_tpr, color='teal',
               label=r'LDICDL = %0.4f' % (LDICDLroc_auc),
               lw=2, alpha=.8)


ax.legend(loc="lower right")

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],xlabel="True Positive Rate",ylabel="False Positive Rate",
       title="Receiver operating characteristic example")
plt.savefig('rocDataset1_5cv_comparison_0718.png', dpi=600)
plt.show()


#  PR
mean_precision, mean_recall, _ = metrics.precision_recall_curve(y_real, y_proba)
mean_aupr = auc(mean_recall, mean_precision)
p1, = plt.plot(mean_recall, mean_precision, label=r'GCMCLDA = %0.4f' % (mean_aupr),
         lw=2, color='r', alpha=.8)

p2, = plt.plot(LDICDLmean_recall, LDICDLmean_precision, label=r'LDICDL = %0.4f' % (LDICDLroc_aupr),
         lw=2, alpha=.8)

p3, = plt.plot(GAMCLDA_R, GAMCLDA_P, label=r'GAMCLDA = %0.4f' % (GAMCLDA_meanaupr),
         lw=2, alpha=.8)


plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.title('Precision/Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="upper right")
plt.savefig('prDataset1_5cv_comparison_0718.png', dpi=600)
plt.show()