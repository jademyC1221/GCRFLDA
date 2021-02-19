# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 14:13:39 2021

@author: Chenmeijun
"""

import Dataset4_1 

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from scipy import interp
#import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def to_torch_sparse_tensor(x):
    if not sp.isspmatrix_coo(x):
        x = sp.coo_matrix(x)
    row, col = x.row, x.col
#    data = x.data 
    indices = torch.from_numpy(np.asarray([row, col]).astype('int64')).long()
    values = torch.from_numpy(x.data.astype(np.float32))
    th_sparse_tensor = torch.sparse.FloatTensor(indices, values, 
                                                x.shape)
    return th_sparse_tensor
    

def tensor_from_numpy(x):
    
    return torch.from_numpy(x)



#读取数据
import pickle
pkl_file=open('LDNFSGBcase10cv.pkl','rb')
U2madj =pickle.load(pkl_file)
M2uadj=pickle.load(pkl_file)
Uidf=pickle.load(pkl_file)
Midf=pickle.load(pkl_file)
Uind=pickle.load(pkl_file)
Mind=pickle.load(pkl_file)
Label=pickle.load(pkl_file)
Tramask=pickle.load(pkl_file)
Testmask=pickle.load(pkl_file)
pkl_file.close()


user2movie_adjacencies = U2madj[0]
user2movie_adjacencies2 = U2madj[1]
user2movie_adjacencies3 = U2madj[2]
user2movie_adjacencies4 = U2madj[3]
user2movie_adjacencies5 = U2madj[4]
user2movie_adjacencies6 = U2madj[5]
user2movie_adjacencies7 = U2madj[6]
user2movie_adjacencies8 = U2madj[7]
user2movie_adjacencies9 = U2madj[8]
user2movie_adjacencies10 = U2madj[9]


movie2user_adjacencies = M2uadj[0]
movie2user_adjacencies2 = M2uadj[1]
movie2user_adjacencies3 = M2uadj[2]
movie2user_adjacencies4 = M2uadj[3]
movie2user_adjacencies5 = M2uadj[4]
movie2user_adjacencies6 = M2uadj[5]
movie2user_adjacencies7 = M2uadj[6]
movie2user_adjacencies8 = M2uadj[7]
movie2user_adjacencies9 = M2uadj[8]
movie2user_adjacencies10 = M2uadj[9]

user_identity_feature = Uidf[0]
user_identity_feature2 = Uidf[1]
user_identity_feature3 = Uidf[2]
user_identity_feature4 = Uidf[3]
user_identity_feature5 = Uidf[4]
user_identity_feature6 = Uidf[5]
user_identity_feature7 = Uidf[6]
user_identity_feature8 = Uidf[7]
user_identity_feature9 = Uidf[8]
user_identity_feature10 = Uidf[9]


movie_indentity_feature = Midf[0]
movie_indentity_feature2 = Midf[1]
movie_indentity_feature3 = Midf[2]
movie_indentity_feature4 = Midf[3]
movie_indentity_feature5 = Midf[4]
movie_indentity_feature6 = Midf[5]
movie_indentity_feature7 = Midf[6]
movie_indentity_feature8 = Midf[7]
movie_indentity_feature9 = Midf[8]
movie_indentity_feature10 = Midf[9]

user_indices=Uind[0]
user_indices2=Uind[1]
user_indices3=Uind[2]
user_indices4=Uind[3]
user_indices5=Uind[4]
user_indices6=Uind[5]
user_indices7=Uind[6]
user_indices8=Uind[7]
user_indices9=Uind[8]
user_indices10=Uind[9]

movie_indices=Mind[0]
movie_indices2=Mind[1]
movie_indices3=Mind[2]
movie_indices4=Mind[3]
movie_indices5=Mind[4]
movie_indices6=Mind[5]
movie_indices7=Mind[6]
movie_indices8=Mind[7]
movie_indices9=Mind[8]
movie_indices10=Mind[9]

labels=Label[0]
labels2=Label[1]
labels3=Label[2]
labels4=Label[3]
labels5=Label[4]
labels6=Label[5]
labels7=Label[6]
labels8=Label[7]
labels9=Label[8]
labels10=Label[9]

train_mask=Tramask[0]
train_mask2=Tramask[1]
train_mask3=Tramask[2]
train_mask4=Tramask[3]
train_mask5=Tramask[4]
train_mask6=Tramask[5]
train_mask7=Tramask[6]
train_mask8=Tramask[7]
train_mask9=Tramask[8]
train_mask10=Tramask[9]

test_mask=Testmask[0]
test_mask2=Testmask[1]
test_mask3=Testmask[2]
test_mask4=Testmask[3]
test_mask5=Testmask[4]
test_mask6=Testmask[5]
test_mask7=Testmask[6]
test_mask8=Testmask[7]
test_mask9=Testmask[8]
test_mask10=Testmask[9]

user2movie_adjacencies = [to_torch_sparse_tensor(adj) for adj in user2movie_adjacencies]
movie2user_adjacencies = [to_torch_sparse_tensor(adj) for adj in movie2user_adjacencies]
user_identity_feature  = tensor_from_numpy(user_identity_feature).float()
movie_identity_feature = tensor_from_numpy(movie_indentity_feature).float()
user_indices           = tensor_from_numpy(user_indices).long()
movie_indices          = tensor_from_numpy(movie_indices).long()
labels1                 = tensor_from_numpy(labels)
train_mask1             = tensor_from_numpy(train_mask)
test_mask1              = tensor_from_numpy(test_mask)

user2movie_adjacencies2 = [to_torch_sparse_tensor(adj) for adj in user2movie_adjacencies2]
movie2user_adjacencies2 = [to_torch_sparse_tensor(adj) for adj in movie2user_adjacencies2]
user_identity_feature2  = tensor_from_numpy(user_identity_feature2).float()
movie_identity_feature2 = tensor_from_numpy(movie_indentity_feature2).float()
user_indices2           = tensor_from_numpy(user_indices2).long()
movie_indices2          = tensor_from_numpy(movie_indices2).long()
labels2                 = tensor_from_numpy(labels2)
train_mask2             = tensor_from_numpy(train_mask2)
test_mask2              = tensor_from_numpy(test_mask2)


user2movie_adjacencies3 = [to_torch_sparse_tensor(adj) for adj in user2movie_adjacencies3]
movie2user_adjacencies3 = [to_torch_sparse_tensor(adj) for adj in movie2user_adjacencies3]
user_identity_feature3  = tensor_from_numpy(user_identity_feature3).float()
movie_identity_feature3 = tensor_from_numpy(movie_indentity_feature3).float()
user_indices3           = tensor_from_numpy(user_indices3).long()
movie_indices3          = tensor_from_numpy(movie_indices3).long()
labels3                 = tensor_from_numpy(labels3)
train_mask3             = tensor_from_numpy(train_mask3)
test_mask3              = tensor_from_numpy(test_mask3)


user2movie_adjacencies4 = [to_torch_sparse_tensor(adj) for adj in user2movie_adjacencies4]
movie2user_adjacencies4 = [to_torch_sparse_tensor(adj) for adj in movie2user_adjacencies4]
user_identity_feature4  = tensor_from_numpy(user_identity_feature4).float()
movie_identity_feature4 = tensor_from_numpy(movie_indentity_feature4).float()
user_indices4           = tensor_from_numpy(user_indices4).long()
movie_indices4          = tensor_from_numpy(movie_indices4).long()
labels4                 = tensor_from_numpy(labels4)
train_mask4             = tensor_from_numpy(train_mask4)
test_mask4              = tensor_from_numpy(test_mask4)


user2movie_adjacencies5 = [to_torch_sparse_tensor(adj) for adj in user2movie_adjacencies5]
movie2user_adjacencies5 = [to_torch_sparse_tensor(adj) for adj in movie2user_adjacencies5]
user_identity_feature5  = tensor_from_numpy(user_identity_feature5).float()
movie_identity_feature5 = tensor_from_numpy(movie_indentity_feature5).float()
user_indices5           = tensor_from_numpy(user_indices5).long()
movie_indices5          = tensor_from_numpy(movie_indices5).long()
labels5                 = tensor_from_numpy(labels5)
train_mask5             = tensor_from_numpy(train_mask5)
test_mask5              = tensor_from_numpy(test_mask5)

user2movie_adjacencies6 = [to_torch_sparse_tensor(adj) for adj in user2movie_adjacencies6]
movie2user_adjacencies6 = [to_torch_sparse_tensor(adj) for adj in movie2user_adjacencies6]
user_identity_feature6  = tensor_from_numpy(user_identity_feature6).float()
movie_identity_feature6 = tensor_from_numpy(movie_indentity_feature6).float()
user_indices6           = tensor_from_numpy(user_indices6).long()
movie_indices6          = tensor_from_numpy(movie_indices6).long()
labels6                 = tensor_from_numpy(labels6)
train_mask6             = tensor_from_numpy(train_mask6)
test_mask6              = tensor_from_numpy(test_mask6)

user2movie_adjacencies7 = [to_torch_sparse_tensor(adj) for adj in user2movie_adjacencies7]
movie2user_adjacencies7 = [to_torch_sparse_tensor(adj) for adj in movie2user_adjacencies7]
user_identity_feature7  = tensor_from_numpy(user_identity_feature7).float()
movie_identity_feature7 = tensor_from_numpy(movie_indentity_feature7).float()
user_indices7           = tensor_from_numpy(user_indices7).long()
movie_indices7          = tensor_from_numpy(movie_indices7).long()
labels7                 = tensor_from_numpy(labels7)
train_mask7             = tensor_from_numpy(train_mask7)
test_mask7              = tensor_from_numpy(test_mask7)

user2movie_adjacencies8 = [to_torch_sparse_tensor(adj) for adj in user2movie_adjacencies8]
movie2user_adjacencies8 = [to_torch_sparse_tensor(adj) for adj in movie2user_adjacencies8]
user_identity_feature8  = tensor_from_numpy(user_identity_feature8).float()
movie_identity_feature8 = tensor_from_numpy(movie_indentity_feature8).float()
user_indices8           = tensor_from_numpy(user_indices8).long()
movie_indices8          = tensor_from_numpy(movie_indices8).long()
labels8                 = tensor_from_numpy(labels8)
train_mask8             = tensor_from_numpy(train_mask8)
test_mask8              = tensor_from_numpy(test_mask8)

user2movie_adjacencies9 = [to_torch_sparse_tensor(adj) for adj in user2movie_adjacencies9]
movie2user_adjacencies9 = [to_torch_sparse_tensor(adj) for adj in movie2user_adjacencies9]
user_identity_feature9  = tensor_from_numpy(user_identity_feature9).float()
movie_identity_feature9 = tensor_from_numpy(movie_indentity_feature9).float()
user_indices9           = tensor_from_numpy(user_indices9).long()
movie_indices9          = tensor_from_numpy(movie_indices9).long()
labels9                 = tensor_from_numpy(labels9)
train_mask9             = tensor_from_numpy(train_mask9)
test_mask9              = tensor_from_numpy(test_mask9)

user2movie_adjacencies10 = [to_torch_sparse_tensor(adj) for adj in user2movie_adjacencies10]
movie2user_adjacencies10 = [to_torch_sparse_tensor(adj) for adj in movie2user_adjacencies10]
user_identity_feature10  = tensor_from_numpy(user_identity_feature10).float()
movie_identity_feature10 = tensor_from_numpy(movie_indentity_feature10).float()
user_indices10           = tensor_from_numpy(user_indices10).long()
movie_indices10          = tensor_from_numpy(movie_indices10).long()
labels10                 = tensor_from_numpy(labels10)
train_mask10             = tensor_from_numpy(train_mask10)
test_mask10              = tensor_from_numpy(test_mask10)


num_user = len(user_identity_feature)    #lnc
num_movie = len(movie_indentity_feature)  #dis

NUM_BASIS = 2
WEIGHT_DACAY = 0.
NODE_INPUT_DIM = user_identity_feature.shape[1]  #节点总数量

EPOCHS = 180       
DROPOUT_RATIO = 0.75    #0.25  0.5
LEARNING_RATE = 0.001   #学习率1e-2=0.01,0.001
GCN_HIDDEN_DIM = 500    #GCN隐藏层
SIDE_HIDDEN_DIM = 10    #边隐藏层
ENCODE_HIDDEN_DIM = 50  #编码隐藏层 
mean_fpr = np.linspace(0, 1, 100)
AUCresult=[]
AUPRresult=[]
Weight=[]
    
from scipy.io import loadmat
datadict = loadmat("E2.mat")  
weightslist = datadict.get('E2') 

for iii in range(0,9,1):   
    lncfilename="ldnfsgb_lncsim"+weightslist[iii]
    for jjj in range(0,9,1):
        disfilename="ldnfsgb_dissim"+weightslist[jjj]

        lncfile="./LDNFSGBsimlkf/" +lncfilename
        disfile="./LDNFSGBsimlkf/" +disfilename

        users_df, movie_df = Dataset4_1.read_data(lncfile,disfile,lncfilename,disfilename)
        SIDE_FEATURE_DIM = users_df.shape[1]+movie_df.shape[1]

        user_side_feature, movie_side_feature = Dataset4_1.build_graph_df(lnclnc=users_df, disdis=movie_df, symmetric_normalization=False)
        user_side_feature      = tensor_from_numpy(user_side_feature).float()
        movie_side_feature     = tensor_from_numpy(movie_side_feature).float()


        ######hyper
        model_inputs1 = (user2movie_adjacencies, movie2user_adjacencies,
                            user_identity_feature, movie_identity_feature,
                            user_side_feature, movie_side_feature, user_indices, movie_indices)
        model_inputs2 = (user2movie_adjacencies2, movie2user_adjacencies2,
                            user_identity_feature2, movie_identity_feature2,
                            user_side_feature, movie_side_feature, user_indices2, movie_indices2)
        model_inputs3 = (user2movie_adjacencies3, movie2user_adjacencies3,
                            user_identity_feature3, movie_identity_feature3,
                            user_side_feature, movie_side_feature, user_indices3, movie_indices3)
        model_inputs4 = (user2movie_adjacencies4, movie2user_adjacencies4,
                            user_identity_feature4, movie_identity_feature4,
                            user_side_feature, movie_side_feature, user_indices4, movie_indices4)
        model_inputs5 = (user2movie_adjacencies5, movie2user_adjacencies5,
                            user_identity_feature5, movie_identity_feature5,
                            user_side_feature, movie_side_feature, user_indices5, movie_indices5)
        model_inputs6 = (user2movie_adjacencies6, movie2user_adjacencies6,
                            user_identity_feature6, movie_identity_feature6,
                            user_side_feature, movie_side_feature, user_indices6, movie_indices6)
        model_inputs7 = (user2movie_adjacencies7, movie2user_adjacencies7,
                            user_identity_feature7, movie_identity_feature7,
                            user_side_feature, movie_side_feature, user_indices7, movie_indices7)
        model_inputs8 = (user2movie_adjacencies8, movie2user_adjacencies8,
                            user_identity_feature8, movie_identity_feature8,
                            user_side_feature, movie_side_feature, user_indices8, movie_indices8)
        model_inputs9 = (user2movie_adjacencies9, movie2user_adjacencies9,
                            user_identity_feature9, movie_identity_feature9,
                            user_side_feature, movie_side_feature, user_indices9, movie_indices9)
        model_inputs10 = (user2movie_adjacencies10, movie2user_adjacencies10,
                            user_identity_feature10, movie_identity_feature10,
                            user_side_feature, movie_side_feature, user_indices10, movie_indices10)
    
        tprs = []
        aucs = []
        auprcs=[]
    #    losstrain_5fold=[]
    #    train_acc_5fold = []
    #    losstest_history = []
    #    test_acc_history = []
        testprob =  []
        testpred_y = []
        y_real = []
        
        for i in range(10):
    
            model = Dataset4_1.GraphMatrixCompletion(NODE_INPUT_DIM, SIDE_FEATURE_DIM, GCN_HIDDEN_DIM,
                                    SIDE_HIDDEN_DIM, ENCODE_HIDDEN_DIM, num_basis=NUM_BASIS)
        
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DACAY)
        
            if i==0:
                model_inputs = model_inputs1
                train_mask = train_mask1
                test_mask = test_mask1
                labels = labels1
            elif i==1:
                model_inputs = model_inputs2
                train_mask = train_mask2
                test_mask = test_mask2
                labels = labels2
            elif i==2:
                model_inputs = model_inputs3
                train_mask = train_mask3
                test_mask = test_mask3
                labels = labels3
            elif i==3:
                model_inputs = model_inputs4
                train_mask = train_mask4
                test_mask = test_mask4
                labels = labels4
            elif i==4 :
                model_inputs = model_inputs5
                train_mask = train_mask5
                test_mask = test_mask5
                labels = labels5
            elif i==5 :
                model_inputs = model_inputs6
                train_mask = train_mask6
                test_mask = test_mask6
                labels = labels6
            elif i==6 :
                model_inputs = model_inputs7
                train_mask = train_mask7
                test_mask = test_mask7
                labels = labels7
            elif i==7 :
                model_inputs = model_inputs8
                train_mask = train_mask8
                test_mask = test_mask8
                labels = labels8
            elif i==8 :
                model_inputs = model_inputs9
                train_mask = train_mask9
                test_mask = test_mask9
                labels = labels9
            else :
                model_inputs = model_inputs10
                train_mask = train_mask10
                test_mask = test_mask10
                labels = labels10
    
#            losstrain_history = []
#            train_acc_history = []
            
            model.train()
            for e in range(EPOCHS):
    
                logits = model(*model_inputs)
                loss = criterion(logits[train_mask], labels[train_mask])
    
                optimizer.zero_grad()
                loss.backward()   # 反向传播计算参数的梯度
                optimizer.step()  # 使用优化方法进行梯度更新
        
                train_acc = Dataset4_1.accu(train_mask)
#                losstrain_history = losstrain_history+[loss.item()]
#                train_acc_history = train_acc_history+[train_acc.item()]
                print("Epoch {:03d}: Loss: {:.4f}, TrainAcc {:.4}".format(e, loss.item(), train_acc.item()))
            
#            losstrain_5fold.append(losstrain_history)
#            train_acc_5fold.append(train_acc_history)
            
            model.eval()
            with torch.no_grad():
    
                logits = model(*model_inputs)
        
                loss = criterion(logits[test_mask], labels[test_mask])
                true_y = labels[test_mask]
                prob = F.softmax(logits[test_mask], dim=1)[:,1]
                pred_y = torch.max(F.softmax(logits[test_mask],dim=1), 1)[1].int()
                testprob.append(prob)
                testpred_y.append(pred_y)
                
                fpr, tpr, thresholds = roc_curve(true_y, prob)
                tprs.append(interp(mean_fpr, fpr, tpr))
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)
                
                precision, recall, thresholds = metrics.precision_recall_curve(true_y, prob)
    #            prs.append(interp(mean_fpr, fpr, tpr))
                y_real.append(true_y)
                auprc = auc(recall, precision)
                auprcs.append(auprc)
                
    #            tprs[-1][0]=0.0
    #            #画图，只需要plt.plot(fpr,tpr)
    #            plt.plot(fpr,tpr,lw=1,alpha=0.3,label='ROC fold %d(area=%0.2f)'% (i,roc_auc))
    
                test_acc = Dataset4_1.accu(test_mask)
                lossitem=loss.item()
                test_accitem = test_acc.item()
                print('test on..............lnciii=%d,disjjj=%d,i=%d' % (iii,jjj,i))
#                    print('test on....................%d' % i)
                print('Test On Epoch {}: loss: {:.4f}, TestAcc {:.4}, Testauc {:.4}, Testaupr {:.4},'.format(e, loss.item(), test_acc.item(), roc_auc, auprc))
              
#                losstest_history.append(lossitem)
#                test_acc_history.append(test_accitem)
                model.train()
            del model
        
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        std_tpr = np.std(tprs, axis=0)
        y_real = np.concatenate(y_real)
        y_proba = np.concatenate(testprob)
        mean_precision, mean_recall, _ = metrics.precision_recall_curve(y_real, y_proba)
        mean_aupr = auc(mean_recall, mean_precision)
    
        weight=[weightslist[iii],weightslist[jjj]]
        Weight.append(weight)
        AUCresult.append(mean_auc)
        AUPRresult.append(mean_aupr)
        
        print('Test On iii {} jjj{}: meanauc {:.4}, meanaupr {:.4},'.format(iii,jjj, mean_auc, mean_aupr))
        
        #保存数据
#        import pickle
        output = open('LDNFSGB_casestudy10cv_weightzuhe.pkl','wb')
        pickle.dump(Weight,output)
        pickle.dump(AUCresult,output)
        pickle.dump(AUPRresult,output)
        output.close()

##读取数据
#import pickle
#pkl_file=open('LDNFSGB_casestudy10cv_weightzuhe.pkl','rb')
#Weight =pickle.load(pkl_file)
#AUCresult =pickle.load(pkl_file)
#AUPRresult =pickle.load(pkl_file)
#pkl_file.close()