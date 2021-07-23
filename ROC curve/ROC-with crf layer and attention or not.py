# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 14:21:23 2021

@author: chenmeijun
"""


from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import numpy as np
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

# In[]  Dataset 1

#读取数据
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



import pickle
pkl_file=open('train2697fivefold_ONE_epoch240_cos0.pkl','rb')
p2697ONE_cos1_tprs =pickle.load(pkl_file)
p2697ONE_cos1_mean_auc =pickle.load(pkl_file)
p2697ONE_cos1_std_auc =pickle.load(pkl_file)
p2697ONE_cos1_std_tpr =pickle.load(pkl_file)
p2697ONE_cos1_y_real =pickle.load(pkl_file)
p2697ONE_cos1_y_proba =pickle.load(pkl_file)
p2697ONE_cos1_mean_aupr =pickle.load(pkl_file)
pkl_file.close()


import pickle
pkl_file=open('train2697fivefold_ONE_epoch240_gssim1.pkl','rb')
p2697ONE_gssim1_tprs =pickle.load(pkl_file)
p2697ONE_gssim1_mean_auc =pickle.load(pkl_file)
p2697ONE_gssim1_std_auc =pickle.load(pkl_file)
p2697ONE_gssim1_std_tpr =pickle.load(pkl_file)
p2697ONE_gssim1_y_real =pickle.load(pkl_file)
p2697ONE_gssim1_y_proba =pickle.load(pkl_file)
p2697ONE_gssim1_mean_aupr =pickle.load(pkl_file)
pkl_file.close()



import pickle
pkl_file=open('train2697fivefold_NOcrf_1928_epoch2402.pkl','rb')
p2697NOcrf_tprs =pickle.load(pkl_file)
p2697NOcrf_mean_auc =pickle.load(pkl_file)
p2697NOcrf_std_auc =pickle.load(pkl_file)
p2697NOcrf_std_tpr =pickle.load(pkl_file)
p2697NOcrf_y_real =pickle.load(pkl_file)
p2697NOcrf_y_proba =pickle.load(pkl_file)
p2697NOcrf_mean_aupr =pickle.load(pkl_file)
pkl_file.close()

import pickle
pkl_file=open('train2697fivefold_Noattention_1928_240_2.pkl','rb')
p2697NOatt_tprs = pickle.load(pkl_file)
p2697NOatt_mean_auc = pickle.load(pkl_file)
p2697NOatt_std_auc = pickle.load(pkl_file)
p2697NOatt_std_tpr = pickle.load(pkl_file)
p2697NOatt_y_real = pickle.load(pkl_file)
p2697NOatt_y_proba = pickle.load(pkl_file)
p2697NOatt_mean_aupr = pickle.load(pkl_file)
pkl_file.close()

#  ROC1-1 
mean_fpr = np.linspace(0, 1, 100)
GCMC_mean_tpr = np.mean(GCMC_tprs, axis=0)
GCMC_mean_tpr[-1] = 1.0
p2697NOcrf_mean_tpr = np.mean(p2697NOcrf_tprs, axis=0)
p2697NOcrf_mean_tpr[-1] = 1.0
p2697NOatt_mean_tpr = np.mean(p2697NOatt_tprs, axis=0)
p2697NOatt_mean_tpr[-1] = 1.0

fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
          alpha=.8)

ax.plot(mean_fpr, GCMC_mean_tpr, color='b',
          label=r'GCRFLDA (AUC = %0.4f $\pm$ %0.4f)' % (GCMC_mean_auc,GCMC_std_auc),
          lw=2, alpha=.8)


ax.plot(mean_fpr, p2697NOatt_mean_tpr, color='r',
          label=r'GCRFLDA without attention (AUC = %0.4f $\pm$ %0.4f)' % (p2697NOatt_mean_auc,p2697NOatt_std_auc),
          lw=2, alpha=.8)

ax.plot(mean_fpr, p2697NOcrf_mean_tpr, color='green',
          label=r'GCRFLDA without CRF layer (AUC = %0.4f $\pm$ %0.4f)' % (p2697NOcrf_mean_auc,p2697NOcrf_std_auc),
          lw=2, alpha=.8)

ax.legend(loc="lower right")
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],xlabel="True Positive Rate",ylabel="False Positive Rate")

plt.savefig('Dataset1_crfatt_ROC01.png', dpi=600)
plt.show()


#  ROC1-2
mean_fpr = np.linspace(0, 1, 100)
GCMC_mean_tpr = np.mean(GCMC_tprs, axis=0)
GCMC_mean_tpr[-1] = 1.0
p2697ONE_cos1_mean_tpr = np.mean(p2697ONE_cos1_tprs, axis=0)
p2697ONE_cos1_mean_tpr[-1] = 1.0
p2697ONE_gssim1_mean_tpr = np.mean(p2697ONE_gssim1_tprs, axis=0)
p2697ONE_gssim1_mean_tpr[-1] = 1.0

fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
          alpha=.8)

ax.plot(mean_fpr, GCMC_mean_tpr, color='b',
          label=r'GCRFLDA with two similarity (AUC = %0.4f $\pm$ %0.4f)' % (GCMC_mean_auc,GCMC_std_auc),
          lw=2, alpha=.8)

ax.plot(mean_fpr, p2697ONE_cos1_mean_tpr,  color='r',
          label=r'GCRFLDA with cosine (AUC = %0.4f $\pm$ %0.4f)' % (p2697ONE_cos1_mean_auc,p2697ONE_cos1_std_auc),
          lw=2, alpha=.8)

ax.plot(mean_fpr, p2697ONE_gssim1_mean_tpr, color='green',
          label=r'GCRFLDA with gaussian (AUC = %0.4f $\pm$ %0.4f)' % (p2697ONE_gssim1_mean_auc,p2697ONE_gssim1_std_auc),
          lw=2, alpha=.8)

ax.legend(loc="lower right")
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],xlabel="True Positive Rate",ylabel="False Positive Rate")

plt.savefig('Dataset1_ONE_ROC01.png', dpi=600)
plt.show()


# In[]   Dataset 2

from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import numpy as np
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)


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

import pickle
pkl_file=open('DMFLDA_2200.750.001_noattention_2.pkl','rb')
DMFLDA_NOatt_19190_tprs =pickle.load(pkl_file)
DMFLDA_NOatt_19190_mean_auc =pickle.load(pkl_file)
DMFLDA_NOatt_19190_std_auc =pickle.load(pkl_file)
DMFLDA_NOatt_19190_std_tpr =pickle.load(pkl_file)
DMFLDA_NOatt_19190_y_real =pickle.load(pkl_file)
DMFLDA_NOatt_19190_y_proba =pickle.load(pkl_file)
DMFLDA_NOatt_19190_mean_aupr =pickle.load(pkl_file)
pkl_file.close()


#  ROC2-1
mean_fpr = np.linspace(0, 1, 100)
DMFLDA_mean_tpr = np.mean(DMFLDA_tprs, axis=0)
DMFLDA_mean_tpr[-1] = 1.0
DMFLDA_NOcrf_19190_mean_tpr = np.mean(DMFLDA_NOcrf_19190_tprs, axis=0)
DMFLDA_NOcrf_19190_mean_tpr[-1] = 1.0
DMFLDA_NOatt_19190_mean_tpr = np.mean(DMFLDA_NOatt_19190_tprs, axis=0)
DMFLDA_NOatt_19190_mean_tpr[-1] = 1.0

fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
          alpha=.8)

ax.plot(mean_fpr, DMFLDA_mean_tpr, color='b',
          label=r'GCRFLDA (AUC = %0.4f $\pm$ %0.4f)' % (DMFLDA_mean_auc,DMFLDA_std_auc),
          lw=2, alpha=.8)

ax.plot(mean_fpr, DMFLDA_NOatt_19190_mean_tpr, color='green',
          label=r'GCRFLDA without attention (AUC = %0.4f $\pm$ %0.4f)' % (DMFLDA_NOatt_19190_mean_auc,DMFLDA_NOatt_19190_std_auc),
          lw=2, alpha=.8)

ax.plot(mean_fpr, DMFLDA_NOcrf_19190_mean_tpr, color='coral',
          label=r'GCRFLDA without CRF layer (AUC = %0.4f $\pm$ %0.4f)' % (DMFLDA_NOcrf_19190_mean_auc,DMFLDA_NOcrf_19190_std_auc),
          lw=2, alpha=.8)



ax.legend(loc="lower right")
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],xlabel="True Positive Rate",ylabel="False Positive Rate")

plt.savefig('Dataset2_crfatt_ROC01.png', dpi=600)
plt.show()



#  ROC2-2
mean_fpr = np.linspace(0, 1, 100)
DMFLDA_mean_tpr = np.mean(DMFLDA_tprs, axis=0)
DMFLDA_mean_tpr[-1] = 1.0
DMFLDA_ONE_cos1_mean_tpr = np.mean(DMFLDA_ONE_cos1_tprs, axis=0)
DMFLDA_ONE_cos1_mean_tpr[-1] = 1.0
DMFLDA_ONE_gssim0_mean_tpr = np.mean(DMFLDA_ONE_gssim0_tprs, axis=0)
DMFLDA_ONE_gssim0_mean_tpr[-1] = 1.0

fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
          alpha=.8)

ax.plot(mean_fpr, DMFLDA_mean_tpr, color='b',
          label=r'GCRFLDA with two similarity (AUC = %0.4f $\pm$ %0.4f)' % (DMFLDA_mean_auc,DMFLDA_std_auc),
          lw=2, alpha=.8)

ax.plot(mean_fpr, DMFLDA_ONE_cos1_mean_tpr, color='r',
          label=r'GCRFLDA with cosine (AUC = %0.4f $\pm$ %0.4f)' % (DMFLDA_ONE_cos1_mean_auc,DMFLDA_ONE_cos1_std_auc),
          lw=2, alpha=.8)

ax.plot(mean_fpr, DMFLDA_ONE_gssim0_mean_tpr, color='green',
          label=r'GCRFLDA with gaussian (AUC = %0.4f $\pm$ %0.4f)' % (DMFLDA_ONE_gssim0_mean_auc,DMFLDA_ONE_gssim0_std_auc),
          lw=2, alpha=.8)


ax.legend(loc="lower right")
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],xlabel="True Positive Rate",ylabel="False Positive Rate")

plt.savefig('Dataset2_ONE_ROC01.png', dpi=600)
plt.show()




# In[]  Dataset 3

from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import numpy as np


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


import pickle
pkl_file=open('LNSUBRW1937_1800.750.001_noattention_1.pkl','rb')
LNSUBRW_NOatt_19372_tprs =pickle.load(pkl_file)
LNSUBRW_NOatt_19372_mean_auc =pickle.load(pkl_file)
LNSUBRW_NOatt_19372_std_auc =pickle.load(pkl_file)
LNSUBRW_NOatt_19372_std_tpr =pickle.load(pkl_file)
LNSUBRW_NOatt_19372_y_real =pickle.load(pkl_file)
LNSUBRW_NOatt_19372_y_proba =pickle.load(pkl_file)
LNSUBRW_NOatt_19372_mean_aupr =pickle.load(pkl_file)
pkl_file.close()


#  ROC3-1
mean_fpr = np.linspace(0, 1, 100)
LNSUBRW_mean_tpr = np.mean(LNSUBRW_tprs, axis=0)
LNSUBRW_mean_tpr[-1] = 1.0
LNSUBRW_NOcrf1_19372_mean_tpr = np.mean(LNSUBRW_NOcrf1_19372_tprs, axis=0)
LNSUBRW_NOcrf1_19372_mean_tpr[-1] = 1.0
LNSUBRW_NOatt_19372_mean_tpr = np.mean(LNSUBRW_NOatt_19372_tprs, axis=0)
LNSUBRW_NOatt_19372_mean_tpr[-1] = 1.0

fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
          alpha=.8)

ax.plot(mean_fpr, LNSUBRW_mean_tpr, color='b',
          label=r'GCRFLDA (AUC = %0.4f $\pm$ %0.4f)' % (LNSUBRW_mean_auc,LNSUBRW_std_auc),
          lw=2, alpha=.8)

ax.plot(mean_fpr, LNSUBRW_NOatt_19372_mean_tpr, color='green',
          label=r'GCRFLDA without attention (AUC = %0.4f $\pm$ %0.4f)' % (LNSUBRW_NOatt_19372_mean_auc,LNSUBRW_NOatt_19372_std_auc),
          lw=2, alpha=.8)

ax.plot(mean_fpr, LNSUBRW_NOcrf1_19372_mean_tpr, color='coral',
          label=r'GCRFLDA without CRF layer (AUC = %0.4f $\pm$ %0.4f)' % (LNSUBRW_NOcrf1_19372_mean_auc,LNSUBRW_NOcrf1_19372_std_auc),
          lw=2, alpha=.8)

ax.legend(loc="lower right")
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],xlabel="True Positive Rate",ylabel="False Positive Rate")

plt.savefig('Dataset3_crfatt_ROC01.png', dpi=600)
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


#  ROC3-2
mean_fpr = np.linspace(0, 1, 100)
LNSUBRW_mean_tpr = np.mean(LNSUBRW_tprs, axis=0)
LNSUBRW_mean_tpr[-1] = 1.0
LNSUBRWONE_cos_1937_1_mean_tpr = np.mean(LNSUBRWONE_cos_1937_1_tprs, axis=0)
LNSUBRWONE_cos_1937_1_mean_tpr[-1] = 1.0
LNSUBRWONE_gssim_1937_0_mean_tpr = np.mean(LNSUBRWONE_gssim_1937_0_tprs, axis=0)
LNSUBRWONE_gssim_1937_0_mean_tpr[-1] = 1.0

fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
          alpha=.8)

ax.plot(mean_fpr, LNSUBRW_mean_tpr, color='b',
          label=r'GCRFLDA with two similarity (AUC = %0.4f $\pm$ %0.4f)' % (LNSUBRW_mean_auc,LNSUBRW_std_auc),
          lw=2, alpha=.8)

ax.plot(mean_fpr, LNSUBRWONE_cos_1937_1_mean_tpr, color='r',
          label=r'GCRFLDA with cosine (AUC = %0.4f $\pm$ %0.4f)' % (LNSUBRWONE_cos_1937_1_mean_auc,LNSUBRWONE_cos_1937_1_std_auc),
          lw=2, alpha=.8)

ax.plot(mean_fpr, LNSUBRWONE_gssim_1937_0_mean_tpr, color='green',
          label=r'GCRFLDA with gaussian (AUC = %0.4f $\pm$ %0.4f)' % (LNSUBRWONE_gssim_1937_0_mean_auc,LNSUBRWONE_gssim_1937_0_std_auc),
          lw=2, alpha=.8)

ax.legend(loc="lower right")
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],xlabel="True Positive Rate",ylabel="False Positive Rate")

plt.savefig('Dataset3_ONE_ROC01.png', dpi=600)
plt.show()




# In[]   Dataset 4 

from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import numpy as np
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
 
 
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

import pickle
pkl_file=open('LDNFSGB5cvallneg_NOcrf_7364_1.pkl','rb')
LDNFSGB_case10cv_NOcrf_4655_0_tprs =pickle.load(pkl_file)
LDNFSGB_case10cv_NOcrf_4655_0_mean_auc =pickle.load(pkl_file)
LDNFSGB_case10cv_NOcrf_4655_0_std_auc =pickle.load(pkl_file)
LDNFSGB_case10cv_NOcrf_4655_0_std_tpr =pickle.load(pkl_file)
LDNFSGB_case10cv_NOcrf_4655_0_y_real =pickle.load(pkl_file)
LDNFSGB_case10cv_NOcrf_4655_0_y_proba =pickle.load(pkl_file)
LDNFSGB_case10cv_NOcrf_4655_0_mean_aupr =pickle.load(pkl_file)
pkl_file.close()

import pickle
pkl_file=open('LDNFSGB5cvallneg2400.50.001_noattetion2.pkl','rb')
LDNFSGB_case10cv_NOatt_4655_0_tprs =pickle.load(pkl_file)
LDNFSGB_case10cv_NOatt_4655_0_mean_auc =pickle.load(pkl_file)
LDNFSGB_case10cv_NOatt_4655_0_std_auc =pickle.load(pkl_file)
LDNFSGB_case10cv_NOatt_4655_0_std_tpr =pickle.load(pkl_file)
LDNFSGB_case10cv_NOatt_4655_0_y_real =pickle.load(pkl_file)
LDNFSGB_case10cv_NOatt_4655_0_y_proba =pickle.load(pkl_file)
LDNFSGB_case10cv_NOatt_4655_0_mean_aupr =pickle.load(pkl_file)
pkl_file.close()

#  ROC4-1
mean_fpr = np.linspace(0, 1, 100)
LDNFSGB_mean_tpr = np.mean(LDNFSGB_tprs, axis=0)
LDNFSGB_mean_tpr[-1] = 1.0
LDNFSGB_case10cv_NOcrf_4655_0_mean_tpr = np.mean(LDNFSGB_case10cv_NOcrf_4655_0_tprs, axis=0)
LDNFSGB_case10cv_NOcrf_4655_0_mean_tpr[-1] = 1.0
LDNFSGB_case10cv_NOatt_4655_0_mean_tpr = np.mean(LDNFSGB_case10cv_NOatt_4655_0_tprs, axis=0)
LDNFSGB_case10cv_NOatt_4655_0_mean_tpr[-1] = 1.0


fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
          alpha=.8)

ax.plot(mean_fpr, LDNFSGB_mean_tpr, color='b',
          label=r'GCRFLDA (AUC = %0.4f $\pm$ %0.4f)' % (LDNFSGB_mean_auc,LDNFSGB_std_auc),
          lw=2, alpha=.8)
ax.plot(mean_fpr, LDNFSGB_case10cv_NOatt_4655_0_mean_tpr, color='green',
          label=r'GCRFLDA without attention (AUC = %0.4f $\pm$ %0.4f)' % (LDNFSGB_case10cv_NOatt_4655_0_mean_auc,LDNFSGB_case10cv_NOatt_4655_0_std_auc),
          lw=2, alpha=.8)

ax.plot(mean_fpr, LDNFSGB_case10cv_NOcrf_4655_0_mean_tpr, color='coral',
          label=r'GCRFLDA without CRF layer (AUC = %0.4f $\pm$ %0.4f)' % (LDNFSGB_case10cv_NOcrf_4655_0_mean_auc,LDNFSGB_case10cv_NOcrf_4655_0_std_auc),
          lw=2, alpha=.8)

ax.legend(loc="lower right")
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],xlabel="True Positive Rate",ylabel="False Positive Rate")

plt.savefig('Dataset4_crfatt_ROC01.png', dpi=600)
plt.show()



# ROC4-2

import pickle
pkl_file=open('LDNFSGB5CVallneg_ONE__cos1.pkl','rb')
LDNFSGB_case10cv_ONE__cos0_tprs =pickle.load(pkl_file)
LDNFSGB_case10cv_ONE__cos0_mean_auc =pickle.load(pkl_file)
LDNFSGB_case10cv_ONE__cos0_std_auc =pickle.load(pkl_file)
LDNFSGB_case10cv_ONE__cos0_std_tpr =pickle.load(pkl_file)
LDNFSGB_case10cv_ONE__cos0_y_real =pickle.load(pkl_file)
LDNFSGB_case10cv_ONE__cos0_y_proba =pickle.load(pkl_file)
LDNFSGB_case10cv_ONE__cos0_mean_aupr =pickle.load(pkl_file)
pkl_file.close()

import pickle
pkl_file=open('LDNFSGB5CVallneg_ONE_gssim0.pkl','rb')
LDNFSGB_case10cv_ONE_gssim0_tprs =pickle.load(pkl_file)
LDNFSGB_case10cv_ONE_gssim0_mean_auc =pickle.load(pkl_file)
LDNFSGB_case10cv_ONE_gssim0_std_auc =pickle.load(pkl_file)
LDNFSGB_case10cv_ONE_gssim0_std_tpr =pickle.load(pkl_file)
LDNFSGB_case10cv_ONE_gssim0_y_real =pickle.load(pkl_file)
LDNFSGB_case10cv_ONE_gssim0_y_proba =pickle.load(pkl_file)
LDNFSGB_case10cv_ONE_gssim0_mean_aupr =pickle.load(pkl_file)
pkl_file.close()


#  ROC4-2
mean_fpr = np.linspace(0, 1, 100)
LDNFSGB_mean_tpr = np.mean(LDNFSGB_tprs, axis=0)
LDNFSGB_mean_tpr[-1] = 1.0
LDNFSGB_case10cv_ONE__cos0_mean_tpr = np.mean(LDNFSGB_case10cv_ONE__cos0_tprs, axis=0)
LDNFSGB_case10cv_ONE__cos0_mean_tpr[-1] = 1.0
LDNFSGB_case10cv_ONE_gssim0_mean_tpr = np.mean(LDNFSGB_case10cv_ONE_gssim0_tprs, axis=0)
LDNFSGB_case10cv_ONE_gssim0_mean_tpr[-1] = 1.0

fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
          alpha=.8)

ax.plot(mean_fpr, LDNFSGB_mean_tpr, color='b',
          label=r'GCRFLDA with two similarity (AUC = %0.4f $\pm$ %0.4f)' % (LDNFSGB_mean_auc,LDNFSGB_std_auc),
          lw=2, alpha=.8)

ax.plot(mean_fpr, LDNFSGB_case10cv_ONE__cos0_mean_tpr, color='r',
          label=r'GCRFLDA with cosine (AUC = %0.4f $\pm$ %0.4f)' % (LDNFSGB_case10cv_ONE__cos0_mean_auc,LDNFSGB_case10cv_ONE__cos0_std_auc),
          lw=2, alpha=.8)

ax.plot(mean_fpr, LDNFSGB_case10cv_ONE_gssim0_mean_tpr, color='green',
          label=r'GCRFLDA with gaussian (AUC = %0.4f $\pm$ %0.4f)' % (LDNFSGB_case10cv_ONE_gssim0_mean_auc,LDNFSGB_case10cv_ONE_gssim0_std_auc),
          lw=2, alpha=.8)

ax.legend(loc="lower right")
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],xlabel="True Positive Rate",ylabel="False Positive Rate")

plt.savefig('Dataset4_ONE_ROC01.png', dpi=600)
plt.show()
