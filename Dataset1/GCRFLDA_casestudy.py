#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 01:40:43 2021

@author: lsz
"""

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import scipy.sparse as sp
import GCRFLDA_dataset
import GCRFLDA_model


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



NUM_BASIS = 2
WEIGHT_DACAY = 0.
NODE_INPUT_DIM = 652 
GCN_HIDDEN_DIM = 500   
SIDE_HIDDEN_DIM = 10    
ENCODE_HIDDEN_DIM = 50  

EPOCHS = 220       
DROPOUT_RATIO = 0.5   
LEARNING_RATE = 0.001  


filename='edge2697casestudy.xlsx'
edge_df = GCRFLDA_dataset.read_edge2(filename=filename,sheetName='Sheet 1')
lnc2dis_adjacencies, dis2lnc_adjacencies, \
    lnc_identity_feature, dis_indentity_feature, \
    lnc_indices, dis_indices, labels, train_mask, test_mask = GCRFLDA_dataset.build_graph_adj(edge_df=edge_df, symmetric_normalization=False)



lncfilename="lncsim19"
disfilename="dissim28"

lncfile="./train2697/simlkf2697basedlncdis2/" +lncfilename
disfile="./train2697/simlkf2697basedlncdis2/" +disfilename

users_df, dis_df = GCRFLDA_dataset.read_data(lncfile,disfile,lncfilename,disfilename)
SIDE_FEATURE_DIM = users_df.shape[1]+dis_df.shape[1]

lnc_side_feature, dis_side_feature = GCRFLDA_dataset.build_graph_df(lnclnc=users_df, disdis=dis_df, symmetric_normalization=False)
lnc_side_feature      = tensor_from_numpy(lnc_side_feature).float()
dis_side_feature     = tensor_from_numpy(dis_side_feature).float()


######hyper
lnc2dis_adjacencies = [to_torch_sparse_tensor(adj) for adj in lnc2dis_adjacencies]
dis2lnc_adjacencies = [to_torch_sparse_tensor(adj) for adj in dis2lnc_adjacencies]
lnc_identity_feature  = tensor_from_numpy(lnc_identity_feature).float()
dis_identity_feature = tensor_from_numpy(dis_indentity_feature).float()
lnc_indices           = tensor_from_numpy(lnc_indices).long()
dis_indices          = tensor_from_numpy(dis_indices).long()
labels1                 = tensor_from_numpy(labels)
train_mask1             = tensor_from_numpy(train_mask)
test_mask1              = tensor_from_numpy(test_mask)


num_lnc = len(lnc_identity_feature)    #lnc
num_dis = len(dis_identity_feature)  #dis

model_inputs1 = (lnc2dis_adjacencies, dis2lnc_adjacencies,
                    lnc_identity_feature, dis_identity_feature,
                    lnc_side_feature, dis_side_feature, lnc_indices, dis_indices)


testprob =  []
testpred_y = []
y_real = []
    
model = GraphMatrixCompletion(NODE_INPUT_DIM, SIDE_FEATURE_DIM, GCN_HIDDEN_DIM,
                        SIDE_HIDDEN_DIM, ENCODE_HIDDEN_DIM, num_basis=NUM_BASIS)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DACAY)


model_inputs = model_inputs1
train_mask = train_mask1
test_mask = test_mask1
labels = labels1
   
model.train()
for e in range(EPOCHS):

    logits = model(*model_inputs)
    loss = criterion(logits[train_mask], labels[train_mask])

    optimizer.zero_grad()
    loss.backward()  
    optimizer.step() 

    train_acc = accu(train_mask)
#                losstrain_history = losstrain_history+[loss.item()]
#                train_acc_history = train_acc_history+[train_acc.item()]
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



pred_y0 = np.array(pred_y)
prob0 = np.array(prob)
sum(pred_y0)

#10501/96183
#  Save
import pickle
output = open('case2697_crf_02.pkl','wb')
'''19,28;220,0.5,0.001'''
pickle.dump(edge_df,output)
pickle.dump(pred_y0,output)
pickle.dump(prob0,output)
output.close()

#  Read
import pickle
pkl_file=open('case2697_crf_02.pkl','rb')
'''19,28;220,0.5,0.001'''
edge_df =pickle.load(pkl_file)
pred_y0 =pickle.load(pkl_file)
prob0 =pickle.load(pkl_file)
pkl_file.close()