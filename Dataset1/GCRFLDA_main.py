#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 01:01:31 2021

@author: lsz
"""

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn import metrics
from scipy import interp
from sklearn.metrics import roc_curve, auc
import scipy.sparse as sp

import GCRFLDA_dataset
import GCRFLDA_model


def to_torch_sparse_tensor(x):
    if not sp.isspmatrix_coo(x):
        x = sp.coo_matrix(x)
    row, col = x.row, x.col
    indices = torch.from_numpy(np.asarray([row, col]).astype('int64')).long()
    values = torch.from_numpy(x.data.astype(np.float32))
    th_sparse_tensor = torch.sparse.FloatTensor(indices, values, 
                                                x.shape)
    return th_sparse_tensor
    

def tensor_from_numpy(x):
    
    return torch.from_numpy(x)

class GraphMatrixCompletion(nn.Module):
    def __init__(self, input_dim, side_feat_dim, 
                 gcn_hidden_dim, side_hidden_dim, 
                 encode_hidden_dim, 
                 num_support=2, num_classes=2, num_basis=2):

        super(GraphMatrixCompletion, self).__init__()
        self.encoder = GCRFLDA_model.StackGCNEncoder(input_dim, gcn_hidden_dim, num_support,num_lnc,num_dis)
        self.dense1 = GCRFLDA_model.FullyConnected(side_feat_dim, side_hidden_dim, dropout=0.,
                                     use_bias=True)
        self.dense2 = GCRFLDA_model.FullyConnected(gcn_hidden_dim + side_hidden_dim, encode_hidden_dim,
                                     dropout=DROPOUT_RATIO, activation=lambda x: x)
        self.decoder = GCRFLDA_model.Decoder(encode_hidden_dim, num_basis, num_classes, 
                               dropout=DROPOUT_RATIO, activation=lambda x: x)


    def forward(self, lnc_supports, dis_supports, 
                lnc_inputs, dis_inputs, 
                lnc_side_inputs, dis_side_inputs, 
                lnc_edge_idx, dis_edge_idx):
        lnc_gcn, dis_gcn = self.encoder(lnc_supports, dis_supports, lnc_inputs, dis_inputs) 
        lnc_side_feat, dis_side_feat = self.dense1(lnc_side_inputs, dis_side_inputs)  
        
        lnc_feat = torch.cat((lnc_gcn, lnc_side_feat), dim=1)  
        dis_feat = torch.cat((dis_gcn, dis_side_feat), dim=1)
        
        lnc_embed, dis_embed = self.dense2(lnc_feat, dis_feat)
        
        edge_logits = self.decoder(lnc_embed, dis_embed, lnc_edge_idx, dis_edge_idx)
        
        return edge_logits

# mask_logits[:,1] the second col
def accu(mask):
    model.eval()
    with torch.no_grad():
        logits = model(*model_inputs)
        mask_logits = logits[mask]
        predict_y = mask_logits.max(1)[1] #(1)the max value in each row #[1]the index of the max value
        accuarcy = torch.eq(predict_y, labels[mask]).float().mean()
    return accuarcy

# In[]
    
U2madj=[]
M2uadj=[]
Uidf=[]
Midf=[]
Uind=[]
Mind=[]
Label=[]
Tramask=[]
Testmask=[]
#
for j in range(1,11,1):
#    j=1
    filename='edgf2697tenfold/edge_f10cv'+str(j)+'.xlsx'
    edge_df = GCRFLDA_dataset.read_edge2(filename=filename,sheetName='Sheet 1')
    lnc2dis_adjacencies, dis2lnc_adjacencies, \
        lnc_identity_feature, dis_indentity_feature, \
        lnc_indices, dis_indices, labels, train_mask, test_mask = GCRFLDA_dataset.build_graph_adj(edge_df=edge_df, symmetric_normalization=False)

    edge_df2 = GCRFLDA_dataset.read_edge2(filename=filename,sheetName='Sheet 2')
    lnc2dis_adjacencies2, dis2lnc_adjacencies2, \
        lnc_identity_feature2, dis_indentity_feature2, \
        lnc_indices2, dis_indices2, labels2, train_mask2, test_mask2 = GCRFLDA_dataset.build_graph_adj(edge_df=edge_df2, symmetric_normalization=False)

    edge_df3 = GCRFLDA_dataset.read_edge2(filename=filename,sheetName='Sheet 3')
    lnc2dis_adjacencies3, dis2lnc_adjacencies3, \
        lnc_identity_feature3, dis_indentity_feature3, \
        lnc_indices3, dis_indices3, labels3, train_mask3, test_mask3 = GCRFLDA_dataset.build_graph_adj(edge_df=edge_df3, symmetric_normalization=False)

    edge_df4 = GCRFLDA_dataset.read_edge2(filename=filename,sheetName='Sheet 4')
    lnc2dis_adjacencies4, dis2lnc_adjacencies4, \
        lnc_identity_feature4, dis_indentity_feature4, \
        lnc_indices4, dis_indices4, labels4, train_mask4, test_mask4 = GCRFLDA_dataset.build_graph_adj(edge_df=edge_df4, symmetric_normalization=False)

    edge_df5 = GCRFLDA_dataset.read_edge2(filename=filename,sheetName='Sheet 5')
    lnc2dis_adjacencies5, dis2lnc_adjacencies5, \
        lnc_identity_feature5, dis_indentity_feature5, \
        lnc_indices5, dis_indices5, labels5, train_mask5, test_mask5 = GCRFLDA_dataset.build_graph_adj(edge_df=edge_df5, symmetric_normalization=False)
    
    edge_df6 = GCRFLDA_dataset.read_edge2(filename=filename,sheetName='Sheet 6')
    lnc2dis_adjacencies6, dis2lnc_adjacencies6, \
        lnc_identity_feature6, dis_indentity_feature6, \
        lnc_indices6, dis_indices6, labels6, train_mask6, test_mask6 = GCRFLDA_dataset.build_graph_adj(edge_df=edge_df6, symmetric_normalization=False)

    edge_df7 = GCRFLDA_dataset.read_edge2(filename=filename,sheetName='Sheet 7')
    lnc2dis_adjacencies7, dis2lnc_adjacencies7, \
        lnc_identity_feature7, dis_indentity_feature7, \
        lnc_indices7, dis_indices7, labels7, train_mask7, test_mask7 = GCRFLDA_dataset.build_graph_adj(edge_df=edge_df7, symmetric_normalization=False)

    edge_df8 = GCRFLDA_dataset.read_edge2(filename=filename,sheetName='Sheet 8')
    lnc2dis_adjacencies8, dis2lnc_adjacencies8, \
        lnc_identity_feature8, dis_indentity_feature8, \
        lnc_indices8, dis_indices8, labels8, train_mask8, test_mask8 = GCRFLDA_dataset.build_graph_adj(edge_df=edge_df8, symmetric_normalization=False)

    edge_df9 = GCRFLDA_dataset.read_edge2(filename=filename,sheetName='Sheet 9')
    lnc2dis_adjacencies9, dis2lnc_adjacencies9, \
        lnc_identity_feature9, dis_indentity_feature9, \
        lnc_indices9, dis_indices9, labels9, train_mask9, test_mask9 = GCRFLDA_dataset.build_graph_adj(edge_df=edge_df9, symmetric_normalization=False)

    edge_df10 = GCRFLDA_dataset.read_edge2(filename=filename,sheetName='Sheet 10')
    lnc2dis_adjacencies10, dis2lnc_adjacencies10, \
        lnc_identity_feature10, dis_indentity_feature10, \
        lnc_indices10, dis_indices10, labels10, train_mask10, test_mask10 = GCRFLDA_dataset.build_graph_adj(edge_df=edge_df10, symmetric_normalization=False)
  
    U=[lnc2dis_adjacencies,lnc2dis_adjacencies2,lnc2dis_adjacencies3,lnc2dis_adjacencies4,lnc2dis_adjacencies5,\
       lnc2dis_adjacencies6,lnc2dis_adjacencies7,lnc2dis_adjacencies8,lnc2dis_adjacencies9,lnc2dis_adjacencies10]
    U2madj.append(U)
    M=[dis2lnc_adjacencies,dis2lnc_adjacencies2,dis2lnc_adjacencies3,dis2lnc_adjacencies4,dis2lnc_adjacencies5,\
       dis2lnc_adjacencies6,dis2lnc_adjacencies7,dis2lnc_adjacencies8,dis2lnc_adjacencies9,dis2lnc_adjacencies10]
    M2uadj.append(M)
    Uid=[lnc_identity_feature,lnc_identity_feature2,lnc_identity_feature3,lnc_identity_feature4,lnc_identity_feature5,\
         lnc_identity_feature6,lnc_identity_feature7,lnc_identity_feature8,lnc_identity_feature9,lnc_identity_feature10]
    Uidf.append(Uid)
    Mid=[dis_indentity_feature,dis_indentity_feature2,dis_indentity_feature3,dis_indentity_feature4,dis_indentity_feature5,\
         dis_indentity_feature6,dis_indentity_feature7,dis_indentity_feature8,dis_indentity_feature9,dis_indentity_feature10]
    Midf.append(Mid)
    Uin=[lnc_indices,lnc_indices2,lnc_indices3,lnc_indices4,lnc_indices5,\
         lnc_indices6,lnc_indices7,lnc_indices8,lnc_indices9,lnc_indices10]
    Uind.append(Uin)
    Min=[dis_indices,dis_indices2,dis_indices3,dis_indices4,dis_indices5,\
         dis_indices6,dis_indices7,dis_indices8,dis_indices9,dis_indices10]
    Mind.append(Min)
    L=[labels,labels2,labels3,labels4,labels5,labels6,labels7,labels8,labels9,labels10]
    Label.append(L)
    Tr=[train_mask,train_mask2,train_mask3,train_mask4,train_mask5,train_mask6,train_mask7,train_mask8,train_mask9,train_mask10]
    Tramask.append(Tr)
    Te=[test_mask,test_mask2,test_mask3,test_mask4,test_mask5,test_mask6,test_mask7,test_mask8,test_mask9,test_mask10]
    Testmask.append(Te)
#
#保存数据
import pickle
output = open('egde2697tenfold_data.pkl','wb')
pickle.dump(U2madj,output)
pickle.dump(M2uadj,output)
pickle.dump(Uidf,output)
pickle.dump(Midf,output)
pickle.dump(Uind,output)
pickle.dump(Mind,output)
pickle.dump(Label,output)
pickle.dump(Tramask,output)
pickle.dump(Testmask,output)
output.close()

#
import pickle
pkl_file=open('egde2697tenfold_data.pkl','rb')
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


tt = 2
lnc2dis_adjacencies = U2madj[tt][0]
lnc2dis_adjacencies2 = U2madj[tt][1]
lnc2dis_adjacencies3 = U2madj[tt][2]
lnc2dis_adjacencies4 = U2madj[tt][3]
lnc2dis_adjacencies5 = U2madj[tt][4]
lnc2dis_adjacencies6 = U2madj[tt][5]
lnc2dis_adjacencies7 = U2madj[tt][6]
lnc2dis_adjacencies8 = U2madj[tt][7]
lnc2dis_adjacencies9 = U2madj[tt][8]
lnc2dis_adjacencies10 = U2madj[tt][9]


dis2lnc_adjacencies = M2uadj[tt][0]
dis2lnc_adjacencies2 = M2uadj[tt][1]
dis2lnc_adjacencies3 = M2uadj[tt][2]
dis2lnc_adjacencies4 = M2uadj[tt][3]
dis2lnc_adjacencies5 = M2uadj[tt][4]
dis2lnc_adjacencies6 = M2uadj[tt][5]
dis2lnc_adjacencies7 = M2uadj[tt][6]
dis2lnc_adjacencies8 = M2uadj[tt][7]
dis2lnc_adjacencies9 = M2uadj[tt][8]
dis2lnc_adjacencies10 = M2uadj[tt][9]

lnc_identity_feature = Uidf[tt][0]
lnc_identity_feature2 = Uidf[tt][1]
lnc_identity_feature3 = Uidf[tt][2]
lnc_identity_feature4 = Uidf[tt][3]
lnc_identity_feature5 = Uidf[tt][4]
lnc_identity_feature6 = Uidf[tt][5]
lnc_identity_feature7 = Uidf[tt][6]
lnc_identity_feature8 = Uidf[tt][7]
lnc_identity_feature9 = Uidf[tt][8]
lnc_identity_feature10 = Uidf[tt][9]


dis_indentity_feature = Midf[tt][0]
dis_indentity_feature2 = Midf[tt][1]
dis_indentity_feature3 = Midf[tt][2]
dis_indentity_feature4 = Midf[tt][3]
dis_indentity_feature5 = Midf[tt][4]
dis_indentity_feature6 = Midf[tt][5]
dis_indentity_feature7 = Midf[tt][6]
dis_indentity_feature8 = Midf[tt][7]
dis_indentity_feature9 = Midf[tt][8]
dis_indentity_feature10 = Midf[tt][9]

lnc_indices=Uind[tt][0]
lnc_indices2=Uind[tt][1]
lnc_indices3=Uind[tt][2]
lnc_indices4=Uind[tt][3]
lnc_indices5=Uind[tt][4]
lnc_indices6=Uind[tt][5]
lnc_indices7=Uind[tt][6]
lnc_indices8=Uind[tt][7]
lnc_indices9=Uind[tt][8]
lnc_indices10=Uind[tt][9]

dis_indices=Mind[tt][0]
dis_indices2=Mind[tt][1]
dis_indices3=Mind[tt][2]
dis_indices4=Mind[tt][3]
dis_indices5=Mind[tt][4]
dis_indices6=Mind[tt][5]
dis_indices7=Mind[tt][6]
dis_indices8=Mind[tt][7]
dis_indices9=Mind[tt][8]
dis_indices10=Mind[tt][9]

labels=Label[tt][0]
labels2=Label[tt][1]
labels3=Label[tt][2]
labels4=Label[tt][3]
labels5=Label[tt][4]
labels6=Label[tt][5]
labels7=Label[tt][6]
labels8=Label[tt][7]
labels9=Label[tt][8]
labels10=Label[tt][9]

train_mask=Tramask[tt][0]
train_mask2=Tramask[tt][1]
train_mask3=Tramask[tt][2]
train_mask4=Tramask[tt][3]
train_mask5=Tramask[tt][4]
train_mask6=Tramask[tt][5]
train_mask7=Tramask[tt][6]
train_mask8=Tramask[tt][7]
train_mask9=Tramask[tt][8]
train_mask10=Tramask[tt][9]

test_mask=Testmask[tt][0]
test_mask2=Testmask[tt][1]
test_mask3=Testmask[tt][2]
test_mask4=Testmask[tt][3]
test_mask5=Testmask[tt][4]
test_mask6=Testmask[tt][5]
test_mask7=Testmask[tt][6]
test_mask8=Testmask[tt][7]
test_mask9=Testmask[tt][8]
test_mask10=Testmask[tt][9]

lnc2dis_adjacencies = [to_torch_sparse_tensor(adj) for adj in lnc2dis_adjacencies]
dis2lnc_adjacencies = [to_torch_sparse_tensor(adj) for adj in dis2lnc_adjacencies]
lnc_identity_feature  = tensor_from_numpy(lnc_identity_feature).float()
dis_identity_feature = tensor_from_numpy(dis_indentity_feature).float()
lnc_indices           = tensor_from_numpy(lnc_indices).long()
dis_indices          = tensor_from_numpy(dis_indices).long()
labels1                 = tensor_from_numpy(labels)
train_mask1             = tensor_from_numpy(train_mask)
test_mask1              = tensor_from_numpy(test_mask)

lnc2dis_adjacencies2 = [to_torch_sparse_tensor(adj) for adj in lnc2dis_adjacencies2]
dis2lnc_adjacencies2 = [to_torch_sparse_tensor(adj) for adj in dis2lnc_adjacencies2]
lnc_identity_feature2  = tensor_from_numpy(lnc_identity_feature2).float()
dis_identity_feature2 = tensor_from_numpy(dis_indentity_feature2).float()
lnc_indices2           = tensor_from_numpy(lnc_indices2).long()
dis_indices2          = tensor_from_numpy(dis_indices2).long()
labels2                 = tensor_from_numpy(labels2)
train_mask2             = tensor_from_numpy(train_mask2)
test_mask2              = tensor_from_numpy(test_mask2)


lnc2dis_adjacencies3 = [to_torch_sparse_tensor(adj) for adj in lnc2dis_adjacencies3]
dis2lnc_adjacencies3 = [to_torch_sparse_tensor(adj) for adj in dis2lnc_adjacencies3]
lnc_identity_feature3  = tensor_from_numpy(lnc_identity_feature3).float()
dis_identity_feature3 = tensor_from_numpy(dis_indentity_feature3).float()
lnc_indices3           = tensor_from_numpy(lnc_indices3).long()
dis_indices3          = tensor_from_numpy(dis_indices3).long()
labels3                 = tensor_from_numpy(labels3)
train_mask3             = tensor_from_numpy(train_mask3)
test_mask3              = tensor_from_numpy(test_mask3)


lnc2dis_adjacencies4 = [to_torch_sparse_tensor(adj) for adj in lnc2dis_adjacencies4]
dis2lnc_adjacencies4 = [to_torch_sparse_tensor(adj) for adj in dis2lnc_adjacencies4]
lnc_identity_feature4  = tensor_from_numpy(lnc_identity_feature4).float()
dis_identity_feature4 = tensor_from_numpy(dis_indentity_feature4).float()
lnc_indices4           = tensor_from_numpy(lnc_indices4).long()
dis_indices4          = tensor_from_numpy(dis_indices4).long()
labels4                 = tensor_from_numpy(labels4)
train_mask4             = tensor_from_numpy(train_mask4)
test_mask4              = tensor_from_numpy(test_mask4)


lnc2dis_adjacencies5 = [to_torch_sparse_tensor(adj) for adj in lnc2dis_adjacencies5]
dis2lnc_adjacencies5 = [to_torch_sparse_tensor(adj) for adj in dis2lnc_adjacencies5]
lnc_identity_feature5  = tensor_from_numpy(lnc_identity_feature5).float()
dis_identity_feature5 = tensor_from_numpy(dis_indentity_feature5).float()
lnc_indices5           = tensor_from_numpy(lnc_indices5).long()
dis_indices5          = tensor_from_numpy(dis_indices5).long()
labels5                 = tensor_from_numpy(labels5)
train_mask5             = tensor_from_numpy(train_mask5)
test_mask5              = tensor_from_numpy(test_mask5)

lnc2dis_adjacencies6 = [to_torch_sparse_tensor(adj) for adj in lnc2dis_adjacencies6]
dis2lnc_adjacencies6 = [to_torch_sparse_tensor(adj) for adj in dis2lnc_adjacencies6]
lnc_identity_feature6  = tensor_from_numpy(lnc_identity_feature6).float()
dis_identity_feature6 = tensor_from_numpy(dis_indentity_feature6).float()
lnc_indices6           = tensor_from_numpy(lnc_indices6).long()
dis_indices6          = tensor_from_numpy(dis_indices6).long()
labels6                 = tensor_from_numpy(labels6)
train_mask6             = tensor_from_numpy(train_mask6)
test_mask6              = tensor_from_numpy(test_mask6)

lnc2dis_adjacencies7 = [to_torch_sparse_tensor(adj) for adj in lnc2dis_adjacencies7]
dis2lnc_adjacencies7 = [to_torch_sparse_tensor(adj) for adj in dis2lnc_adjacencies7]
lnc_identity_feature7  = tensor_from_numpy(lnc_identity_feature7).float()
dis_identity_feature7 = tensor_from_numpy(dis_indentity_feature7).float()
lnc_indices7           = tensor_from_numpy(lnc_indices7).long()
dis_indices7          = tensor_from_numpy(dis_indices7).long()
labels7                 = tensor_from_numpy(labels7)
train_mask7             = tensor_from_numpy(train_mask7)
test_mask7              = tensor_from_numpy(test_mask7)

lnc2dis_adjacencies8 = [to_torch_sparse_tensor(adj) for adj in lnc2dis_adjacencies8]
dis2lnc_adjacencies8 = [to_torch_sparse_tensor(adj) for adj in dis2lnc_adjacencies8]
lnc_identity_feature8  = tensor_from_numpy(lnc_identity_feature8).float()
dis_identity_feature8 = tensor_from_numpy(dis_indentity_feature8).float()
lnc_indices8           = tensor_from_numpy(lnc_indices8).long()
dis_indices8          = tensor_from_numpy(dis_indices8).long()
labels8                 = tensor_from_numpy(labels8)
train_mask8             = tensor_from_numpy(train_mask8)
test_mask8              = tensor_from_numpy(test_mask8)

lnc2dis_adjacencies9 = [to_torch_sparse_tensor(adj) for adj in lnc2dis_adjacencies9]
dis2lnc_adjacencies9 = [to_torch_sparse_tensor(adj) for adj in dis2lnc_adjacencies9]
lnc_identity_feature9  = tensor_from_numpy(lnc_identity_feature9).float()
dis_identity_feature9 = tensor_from_numpy(dis_indentity_feature9).float()
lnc_indices9           = tensor_from_numpy(lnc_indices9).long()
dis_indices9          = tensor_from_numpy(dis_indices9).long()
labels9                 = tensor_from_numpy(labels9)
train_mask9             = tensor_from_numpy(train_mask9)
test_mask9              = tensor_from_numpy(test_mask9)

lnc2dis_adjacencies10 = [to_torch_sparse_tensor(adj) for adj in lnc2dis_adjacencies10]
dis2lnc_adjacencies10 = [to_torch_sparse_tensor(adj) for adj in dis2lnc_adjacencies10]
lnc_identity_feature10  = tensor_from_numpy(lnc_identity_feature10).float()
dis_identity_feature10 = tensor_from_numpy(dis_indentity_feature10).float()
lnc_indices10           = tensor_from_numpy(lnc_indices10).long()
dis_indices10          = tensor_from_numpy(dis_indices10).long()
labels10                 = tensor_from_numpy(labels10)
train_mask10             = tensor_from_numpy(train_mask10)
test_mask10              = tensor_from_numpy(test_mask10)



## 相似性融合权重寻优
from scipy.io import loadmat
datadict = loadmat("E2.mat")  
weightslist = datadict.get('E2') 

num_lnc = len(lnc_identity_feature)    #lnc
num_dis = len(dis_identity_feature)  #dis

NUM_BASIS = 2
WEIGHT_DACAY = 0.
NODE_INPUT_DIM = lnc_identity_feature.shape[1]  #节点总数量(412+240)

GCN_HIDDEN_DIM = 500    
SIDE_HIDDEN_DIM = 10    
ENCODE_HIDDEN_DIM = 50  
mean_fpr = np.linspace(0, 1, 100)

EPOCHS = 180       
DROPOUT_RATIO = 0.75    
LEARNING_RATE = 0.001 
AUCresult=[]
AUPRresult=[]
Weight=[]

for iii in range(0,9,1):   
    lncfilename="lncsim"+weightslist[iii]
    for jjj in range(0,9,1):
        disfilename="dissim"+weightslist[jjj]

        lncfile="./simlkf2697basedlncdis2/" +lncfilename
        disfile="./simlkf2697basedlncdis2/" +disfilename

        users_df, dis_df = GCRFLDA_dataset.read_data(lncfile,disfile,lncfilename,disfilename)
        SIDE_FEATURE_DIM = users_df.shape[1]+dis_df.shape[1]

        lnc_side_feature, dis_side_feature = GCRFLDA_dataset.build_graph_df(lnclnc=users_df, disdis=dis_df, symmetric_normalization=False)
        lnc_side_feature      = tensor_from_numpy(lnc_side_feature).float()
        dis_side_feature     = tensor_from_numpy(dis_side_feature).float()
            
        model_inputs1 = (lnc2dis_adjacencies, dis2lnc_adjacencies,
                            lnc_identity_feature, dis_identity_feature,
                            lnc_side_feature, dis_side_feature, lnc_indices, dis_indices)
        model_inputs2 = (lnc2dis_adjacencies2, dis2lnc_adjacencies2,
                            lnc_identity_feature2, dis_identity_feature2,
                            lnc_side_feature, dis_side_feature, lnc_indices2, dis_indices2)
        model_inputs3 = (lnc2dis_adjacencies3, dis2lnc_adjacencies3,
                            lnc_identity_feature3, dis_identity_feature3,
                            lnc_side_feature, dis_side_feature, lnc_indices3, dis_indices3)
        model_inputs4 = (lnc2dis_adjacencies4, dis2lnc_adjacencies4,
                            lnc_identity_feature4, dis_identity_feature4,
                            lnc_side_feature, dis_side_feature, lnc_indices4, dis_indices4)
        model_inputs5 = (lnc2dis_adjacencies5, dis2lnc_adjacencies5,
                            lnc_identity_feature5, dis_identity_feature5,
                            lnc_side_feature, dis_side_feature, lnc_indices5, dis_indices5)
        model_inputs6 = (lnc2dis_adjacencies6, dis2lnc_adjacencies6,
                            lnc_identity_feature6, dis_identity_feature6,
                            lnc_side_feature, dis_side_feature, lnc_indices6, dis_indices6)
        model_inputs7 = (lnc2dis_adjacencies7, dis2lnc_adjacencies7,
                            lnc_identity_feature7, dis_identity_feature7,
                            lnc_side_feature, dis_side_feature, lnc_indices7, dis_indices7)
        model_inputs8 = (lnc2dis_adjacencies8, dis2lnc_adjacencies8,
                            lnc_identity_feature8, dis_identity_feature8,
                            lnc_side_feature, dis_side_feature, lnc_indices8, dis_indices8)
        model_inputs9 = (lnc2dis_adjacencies9, dis2lnc_adjacencies9,
                            lnc_identity_feature9, dis_identity_feature9,
                            lnc_side_feature, dis_side_feature, lnc_indices9, dis_indices9)
        model_inputs10 = (lnc2dis_adjacencies10, dis2lnc_adjacencies10,
                            lnc_identity_feature10, dis_identity_feature10,
                            lnc_side_feature, dis_side_feature, lnc_indices10, dis_indices10)
    
        tprs = []
        aucs = []
        auprcs=[]
        testprob =  []
        testpred_y = []
        y_real = []
        
        for i in range(10):
    
            model = GraphMatrixCompletion(NODE_INPUT_DIM, SIDE_FEATURE_DIM, GCN_HIDDEN_DIM,
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
            
            model.train()
            for e in range(EPOCHS):
    
                logits = model(*model_inputs)
                loss = criterion(logits[train_mask], labels[train_mask])
    
                optimizer.zero_grad()
                loss.backward()   # 反向传播计算参数的梯度
                optimizer.step()  # 使用优化方法进行梯度更新
        
                train_acc = accu(train_mask)

                print("Epoch {:03d}: Loss: {:.4f}, TrainAcc {:.4}".format(e, loss.item(), train_acc.item()))
            
            
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
    
                test_acc = accu(test_mask)
                lossitem=loss.item()
                test_accitem = test_acc.item()
                print('test on....................iii=%d,jjj=%d,i=%d' % (iii,jjj,i))
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
        import pickle
        output = open('r2697bldsimlncdis2_crffx_weightzuhe_meanpr.pkl','wb')
        pickle.dump(Weight,output)
        pickle.dump(AUCresult,output)
        pickle.dump(AUPRresult,output)
        output.close()

#读取数据
import pickle
pkl_file=open('r2697bldsimlncdis2_crffx_weightzuhe_meanpr.pkl','rb')
Weight =pickle.load(pkl_file)
AUCresult =pickle.load(pkl_file)
AUPRresult =pickle.load(pkl_file)
pkl_file.close()