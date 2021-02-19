# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 14:46:49 2021

@author: Chenmeijun
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp


def globally_normalize_bipartite_adjacency(adjacencies, symmetric=True):
    """ Globally Normalizes set of bipartite adjacency matrices """

    print('{} normalizing bipartite adj'.format(
        ['Asymmetrically', 'Symmetrically'][symmetric]))

    adj_tot = np.sum([adj for adj in adjacencies])
    degree_u = np.asarray(adj_tot.sum(1)).flatten()
    degree_v = np.asarray(adj_tot.sum(0)).flatten()

    # set zeros to inf to avoid dividing by zero
    degree_u[degree_u == 0.] = np.inf
    degree_v[degree_v == 0.] = np.inf

    degree_u_inv_sqrt = 1. / np.sqrt(degree_u)
    degree_v_inv_sqrt = 1. / np.sqrt(degree_v)
    degree_u_inv_sqrt_mat = sp.diags([degree_u_inv_sqrt], [0])
    degree_v_inv_sqrt_mat = sp.diags([degree_v_inv_sqrt], [0])

    degree_u_inv = degree_u_inv_sqrt_mat.dot(degree_u_inv_sqrt_mat) 

    if symmetric:
        adj_norm = [degree_u_inv_sqrt_mat.dot(adj).dot(
            degree_v_inv_sqrt_mat) for adj in adjacencies]

    else:
        adj_norm = [degree_u_inv.dot(adj) for adj in adjacencies]
        
    return adj_norm
#    return degree_u_inv_sqrt_mat,degree_u_inv,adj_norm

#degree_u_inv_sqrt_mat,degree_u_inv,adj_norm = globally_normalize_bipartite_adjacency(user2movie_adjacencies,
#                                                                    symmetric=False)

def get_adjacency(edge_df, num_user, num_movie, symmetric_normalization):
    user2movie_adjacencies = []
    movie2user_adjacencies = []
    train_edge_df = edge_df.loc[edge_df['usage'] == 'train']
    for i in range(2):
        edge_index = train_edge_df.loc[train_edge_df.label == i, ['lncid', 'disid']].to_numpy()
        support = sp.csr_matrix((np.ones(len(edge_index)), (edge_index[:, 0], edge_index[:, 1])),
                                shape=(num_user, num_movie), dtype=np.float32)
        user2movie_adjacencies.append(support)
        movie2user_adjacencies.append(support.T)

    user2movie_adjacencies = globally_normalize_bipartite_adjacency(user2movie_adjacencies,
                                                                    symmetric=symmetric_normalization)
    #943x1628
    movie2user_adjacencies = globally_normalize_bipartite_adjacency(movie2user_adjacencies,
                                                                    symmetric=symmetric_normalization)
    #1628x943

    return user2movie_adjacencies, movie2user_adjacencies


def get_node_identity_feature(num_user, num_movie):
    """one-hot encoding for nodes"""
    identity_feature = np.identity(num_user + num_movie, dtype=np.float32)  #单位阵
    user_identity_feature, movie_indentity_feature = identity_feature[
        :num_user], identity_feature[num_user:]

    return user_identity_feature, movie_indentity_feature

def convert_to_homogeneous(user_feature: np.ndarray, movie_feature: np.ndarray):

    num_user, user_feature_dim = user_feature.shape
    num_movie, movie_feature_dim = movie_feature.shape
    user_feature = np.concatenate(
        [user_feature, np.zeros((num_user, movie_feature_dim))], axis=1) 
    movie_feature = np.concatenate(
        [movie_feature, np.zeros((num_movie, user_feature_dim))], axis=1)

    return user_feature, movie_feature


def read_data(lncfile,disfile,lncfilename,disfilename):

    # item feature
    from scipy.io import loadmat
    lncfeadict = loadmat(lncfile+".mat")
    lncfeature = lncfeadict.get(lncfilename)
    # user feature
    from scipy.io import loadmat
    disfeadict = loadmat(disfile+".mat")
    disfeature = disfeadict.get(disfilename)
    
    return lncfeature, disfeature

def read_edge(filename,sheetName):

    edge_df = pd.read_excel(filename,sheet_name=sheetName,header=0)
    
    return edge_df

def read_edge2(filename,sheetName):

    edge_df = pd.read_excel(filename,sheet_name=sheetName,header=None)
#    columns = list(edge_df.columns)
    columns2 = ['lncnode','disnode','label','lncid','disid','usage']
    for i in range(6):
     edge_df.rename(columns={edge_df.columns[i]: columns2[i]},inplace=True)
    
    edge_df.loc[edge_df['usage']==2222,'usage'] = 'train'
    edge_df.loc[edge_df['usage']==1111,'usage'] = 'test'
    
    return edge_df

#    @staticmethod
def build_graph_adj(edge_df, symmetric_normalization=False):
    
    node_user = edge_df[['lncnode']].drop_duplicates().sort_values('lncnode')
    node_movie = edge_df[['disnode']].drop_duplicates().sort_values('disnode')
    
    num_user = len(node_user)    
    num_movie = len(node_movie) 

    # adjacency
    user2movie_adjacencies, movie2user_adjacencies = get_adjacency(edge_df, num_user, num_movie,
                                                                   symmetric_normalization)

#    user_side_feature, movie_side_feature = convert_to_homogeneous(lnclnc,disdis)

    # one-hot encoding for nodes
    user_identity_feature, movie_indentity_feature = get_node_identity_feature(num_user, num_movie)

    # user_indices, movie_indices, labels, train_mask
    user_indices, movie_indices, labels = edge_df[['lncid', 'disid', 'label']].to_numpy().T
    train_mask = (edge_df['usage'] == 'train').to_numpy()
    test_mask = (edge_df['usage'] == 'test').to_numpy()

    return user2movie_adjacencies, movie2user_adjacencies, \
        user_identity_feature, movie_indentity_feature, \
        user_indices, movie_indices, labels, train_mask, test_mask



def build_graph_df(lnclnc, disdis, symmetric_normalization=False):

    user_side_feature, movie_side_feature = convert_to_homogeneous(lnclnc,disdis)

    return user_side_feature, movie_side_feature


# In[]
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init

class StackGCNEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_support, num_user,num_movie,
                 use_bias=False, activation=F.relu):

        super(StackGCNEncoder, self).__init__()
        self.input_dim = input_dim      
        self.output_dim = output_dim    
        self.num_support = num_support  
        self.use_bias = use_bias
        self.activation = activation
        assert output_dim % num_support == 0
        self.weight = nn.Parameter(torch.Tensor(num_support, 
            input_dim, output_dim // num_support))                   
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim, ))
            self.bias_item = nn.Parameter(torch.Tensor(output_dim, ))
#        self.reset_parameters()
        self.weight = init.kaiming_uniform_(self.weight)  
        
        self.weight_lnc = nn.Parameter(torch.Tensor(1, num_user, num_user))
        self.weight_dis = nn.Parameter(torch.Tensor(1, num_movie, num_movie))
        self.weight_lnc = init.kaiming_uniform_(self.weight_lnc)
        self.weight_dis = init.kaiming_uniform_(self.weight_dis)

    def crf_layer(self,hidden,hidden_new,flag):    
       #
       alpha = 50  
       beta = 1 #50
       
       hidden_extend = torch.from_numpy(hidden).float().unsqueeze(0)  
    
    #   attention
       conv1 = torch.nn.Conv1d(in_channels=250,out_channels=250,kernel_size=1) 
       hidden_extend = hidden_extend.permute(0,2,1) 
       seq_fts = conv1(hidden_extend)                
        
       conv1 = torch.nn.Conv1d(in_channels=250,out_channels=1,kernel_size=1)
       f_1 = conv1(seq_fts)
       conv1 = torch.nn.Conv1d(in_channels=250,out_channels=1,kernel_size=1)
       f_2 = conv1(seq_fts)
       logits = f_1 + f_2.permute(0, 2, 1)               
        
       m = torch.nn.LeakyReLU(0.1)
       if flag==0:
           coefs = torch.nn.functional.softmax(m(logits)+self.weight_lnc)     
       elif flag==1:
           coefs = torch.nn.functional.softmax(m(logits)+self.weight_dis)    
       coefs = coefs[0] 
       
       # fenzi
       coefs = coefs.float()                                 
       hidden_new = torch.from_numpy(hidden_new).float()     
       res = torch.mm(coefs, hidden_new)                     
       hidden_neighbor = torch.mul(res,beta)                  
       hidden = torch.from_numpy(hidden).float()              
       hidden_self = torch.mul(hidden,alpha)                    
       hidden_crf = hidden_neighbor+hidden_self             
       # fenmu
       unit_mat = torch.ones(hidden.shape[0],hidden.shape[1]).float() 
       res = torch.mm(coefs, unit_mat)
       coff_sum = torch.mul(res,beta)
       const = coff_sum + torch.mul(unit_mat,alpha) 
                   #更新后的特征矩阵
       hidden_crf = torch.div(hidden_crf,const)        
       
       return hidden_crf

    def forward(self, user_supports, item_supports, user_inputs, item_inputs):

        assert len(user_supports) == len(item_supports) == self.num_support

        user_hidden = []
        item_hidden = []

        for i in range(self.num_support):

            #特征乘以权重
            tmp_u = torch.matmul(user_inputs, self.weight[i])    
            tmp_v = torch.matmul(item_inputs, self.weight[i])   
			#user邻接矩阵乘以tmp_v（乘以权重之后的特征），得到暂时user隐藏层
            tmp_user_hidden = torch.sparse.mm(user_supports[i], tmp_v)  
            tmp_item_hidden = torch.sparse.mm(item_supports[i], tmp_u) 
            
            hidden_user = tmp_user_hidden.cpu().detach().numpy()    
            hidden_new_user = tmp_user_hidden.cpu().detach().numpy()
            hidden_item = tmp_item_hidden.cpu().detach().numpy()     
            hidden_new_item = tmp_item_hidden.cpu().detach().numpy() 
            
            #The CRF layer
            flag = 0
            user_hidden_crf = self.crf_layer(hidden_user,hidden_new_user,flag) 
            flag = 1
            item_hidden_crf = self.crf_layer(hidden_item,hidden_new_item,flag)
            
#            flag = 0
#            for cv in range(0,1):  
#               user_hidden_crf = self.crf_layer(hidden_user,hidden_new_user,flag)  
#               hidden_new_user = user_hidden_crf
#            flag = 1
#            for cv in range(0,1):  
#               item_hidden_crf = self.crf_layer(hidden_item,hidden_new_item,flag)  
#               hidden_new_item = item_hidden_crf
                        
            user_hidden.append(user_hidden_crf)
            item_hidden.append(item_hidden_crf)

        user_hidden = torch.cat(user_hidden, dim=1)
        item_hidden = torch.cat(item_hidden, dim=1)

        user_outputs = self.activation(user_hidden) 
        item_outputs = self.activation(item_hidden)

        if self.use_bias:
            user_outputs += self.bias
            item_outputs += self.bias_item

        return user_outputs, item_outputs

# In[]: 
        
class FullyConnected(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.,
                 use_bias=False, activation=F.relu,
                 share_weights=False):

        super(FullyConnected, self).__init__()
        self.input_dim = input_dim    
        self.output_dim = output_dim 
        self.use_bias = use_bias
        self.activation = activation
        self.share_weights = share_weights
        self.linear_user = nn.Linear(input_dim, output_dim, bias=use_bias)  
        if self.share_weights:
            self.linear_item = self.linear_user
        else:
            self.linear_item = nn.Linear(input_dim, output_dim, bias=use_bias) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, user_inputs, item_inputs):

        user_inputs = self.dropout(user_inputs)
        user_outputs = self.linear_user(user_inputs)
        
        item_inputs = self.dropout(item_inputs)
        item_outputs = self.linear_item(item_inputs) 

        if self.activation:
            user_outputs = self.activation(user_outputs)  
            item_outputs = self.activation(item_outputs)  

        return user_outputs, item_outputs 

# In[]: 
        
class Decoder(nn.Module):
    def __init__(self, input_dim, num_weights, num_classes, dropout=0., activation=F.relu):

        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.num_weights = num_weights
        self.num_classes = num_classes
        self.activation = activation
        
        self.weight = nn.Parameter(torch.Tensor(num_weights, input_dim, input_dim)) 
        self.weight_classifier = nn.Parameter(torch.Tensor(num_weights, num_classes))
        self.reset_parameters()
        
        self.dropout = nn.Dropout(dropout)  #Dropout(p=0.5)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        init.kaiming_uniform_(self.weight_classifier)

    def forward(self, user_inputs, item_inputs, user_indices, item_indices):
#        edge_logits = self.decoder(user_embed, movie_embed, user_edge_idx, item_edge_idx)

        user_inputs = self.dropout(user_inputs)
        item_inputs = self.dropout(item_inputs)
        user_inputs = user_inputs[user_indices]
        item_inputs = item_inputs[item_indices] 
        
        basis_outputs = []
        for i in range(self.num_weights):
            tmp = torch.matmul(user_inputs, self.weight[i])       
            out = torch.sum(tmp * item_inputs, dim=1, keepdim=True) 
            basis_outputs.append(out)

        basis_outputs = torch.cat(basis_outputs, dim=1) 
        
        outputs = torch.matmul(basis_outputs, self.weight_classifier)
        outputs = self.activation(outputs)
        
        return outputs
    


# In[]

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn import metrics
from scipy import interp
#import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc



class GraphMatrixCompletion(nn.Module):
    def __init__(self, input_dim, side_feat_dim, 
                 gcn_hidden_dim, side_hidden_dim, 
                 encode_hidden_dim, 
                 num_support=2, num_classes=2, num_basis=2):

        super(GraphMatrixCompletion, self).__init__()
        self.encoder = StackGCNEncoder(input_dim, gcn_hidden_dim, num_support,num_user,num_movie,)
        self.dense1 = FullyConnected(side_feat_dim, side_hidden_dim, dropout=0.,
                                     use_bias=True)
        self.dense2 = FullyConnected(gcn_hidden_dim + side_hidden_dim, encode_hidden_dim,
                                     dropout=DROPOUT_RATIO, activation=lambda x: x)
        self.decoder = Decoder(encode_hidden_dim, num_basis, num_classes, 
                               dropout=DROPOUT_RATIO, activation=lambda x: x)


    def forward(self, user_supports, item_supports, 
                user_inputs, item_inputs, 
                user_side_inputs, item_side_inputs, 
                user_edge_idx, item_edge_idx):
        user_gcn, movie_gcn = self.encoder(user_supports, item_supports, user_inputs, item_inputs) 
        user_side_feat, movie_side_feat = self.dense1(user_side_inputs, item_side_inputs)  
        
        user_feat = torch.cat((user_gcn, user_side_feat), dim=1)
        movie_feat = torch.cat((movie_gcn, movie_side_feat), dim=1) 
        
        user_embed, movie_embed = self.dense2(user_feat, movie_feat)
        
        edge_logits = self.decoder(user_embed, movie_embed, user_edge_idx, item_edge_idx)
        
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

#U2madj=[]
#M2uadj=[]
#Uidf=[]
#Midf=[]
#Uind=[]
#Mind=[]
#Label=[]
#Tramask=[]
#Testmask=[]
##
#for j in range(1,11,1):
##    j=1
#    filename='/home/lsz/GraphNeuralNetwork-master/LDNFSGB/LDNFSGBedgecv10/edge_f_ldnfsgb_cv10_'+str(j)+'.xlsx'
#    edge_df = read_edge2(filename=filename,sheetName='Sheet 1')
#    user2movie_adjacencies, movie2user_adjacencies, \
#        user_identity_feature, movie_indentity_feature, \
#        user_indices, movie_indices, labels, train_mask, test_mask = build_graph_adj(edge_df=edge_df, symmetric_normalization=False)
#
#    edge_df2 = read_edge2(filename=filename,sheetName='Sheet 2')
#    user2movie_adjacencies2, movie2user_adjacencies2, \
#        user_identity_feature2, movie_indentity_feature2, \
#        user_indices2, movie_indices2, labels2, train_mask2, test_mask2 = build_graph_adj(edge_df=edge_df2, symmetric_normalization=False)
#
#    edge_df3 = read_edge2(filename=filename,sheetName='Sheet 3')
#    user2movie_adjacencies3, movie2user_adjacencies3, \
#        user_identity_feature3, movie_indentity_feature3, \
#        user_indices3, movie_indices3, labels3, train_mask3, test_mask3 = build_graph_adj(edge_df=edge_df3, symmetric_normalization=False)
#
#    edge_df4 = read_edge2(filename=filename,sheetName='Sheet 4')
#    user2movie_adjacencies4, movie2user_adjacencies4, \
#        user_identity_feature4, movie_indentity_feature4, \
#        user_indices4, movie_indices4, labels4, train_mask4, test_mask4 = build_graph_adj(edge_df=edge_df4, symmetric_normalization=False)
#
#    edge_df5 = read_edge2(filename=filename,sheetName='Sheet 5')
#    user2movie_adjacencies5, movie2user_adjacencies5, \
#        user_identity_feature5, movie_indentity_feature5, \
#        user_indices5, movie_indices5, labels5, train_mask5, test_mask5 = build_graph_adj(edge_df=edge_df5, symmetric_normalization=False)
#    
#    edge_df6 = read_edge2(filename=filename,sheetName='Sheet 6')
#    user2movie_adjacencies6, movie2user_adjacencies6, \
#        user_identity_feature6, movie_indentity_feature6, \
#        user_indices6, movie_indices6, labels6, train_mask6, test_mask6 = build_graph_adj(edge_df=edge_df6, symmetric_normalization=False)
#
#    edge_df7 = read_edge2(filename=filename,sheetName='Sheet 7')
#    user2movie_adjacencies7, movie2user_adjacencies7, \
#        user_identity_feature7, movie_indentity_feature7, \
#        user_indices7, movie_indices7, labels7, train_mask7, test_mask7 = build_graph_adj(edge_df=edge_df7, symmetric_normalization=False)
#
#    edge_df8 = read_edge2(filename=filename,sheetName='Sheet 8')
#    user2movie_adjacencies8, movie2user_adjacencies8, \
#        user_identity_feature8, movie_indentity_feature8, \
#        user_indices8, movie_indices8, labels8, train_mask8, test_mask8 = build_graph_adj(edge_df=edge_df8, symmetric_normalization=False)
#
#    edge_df9 = read_edge2(filename=filename,sheetName='Sheet 9')
#    user2movie_adjacencies9, movie2user_adjacencies9, \
#        user_identity_feature9, movie_indentity_feature9, \
#        user_indices9, movie_indices9, labels9, train_mask9, test_mask9 = build_graph_adj(edge_df=edge_df9, symmetric_normalization=False)
#
#    edge_df10 = read_edge2(filename=filename,sheetName='Sheet 10')
#    user2movie_adjacencies10, movie2user_adjacencies10, \
#        user_identity_feature10, movie_indentity_feature10, \
#        user_indices10, movie_indices10, labels10, train_mask10, test_mask10 = build_graph_adj(edge_df=edge_df10, symmetric_normalization=False)
#  
#    U=[user2movie_adjacencies,user2movie_adjacencies2,user2movie_adjacencies3,user2movie_adjacencies4,user2movie_adjacencies5,\
#       user2movie_adjacencies6,user2movie_adjacencies7,user2movie_adjacencies8,user2movie_adjacencies9,user2movie_adjacencies10]
#    U2madj.append(U)
#    M=[movie2user_adjacencies,movie2user_adjacencies2,movie2user_adjacencies3,movie2user_adjacencies4,movie2user_adjacencies5,\
#       movie2user_adjacencies6,movie2user_adjacencies7,movie2user_adjacencies8,movie2user_adjacencies9,movie2user_adjacencies10]
#    M2uadj.append(M)
#    Uid=[user_identity_feature,user_identity_feature2,user_identity_feature3,user_identity_feature4,user_identity_feature5,\
#         user_identity_feature6,user_identity_feature7,user_identity_feature8,user_identity_feature9,user_identity_feature10]
#    Uidf.append(Uid)
#    Mid=[movie_indentity_feature,movie_indentity_feature2,movie_indentity_feature3,movie_indentity_feature4,movie_indentity_feature5,\
#         movie_indentity_feature6,movie_indentity_feature7,movie_indentity_feature8,movie_indentity_feature9,movie_indentity_feature10]
#    Midf.append(Mid)
#    Uin=[user_indices,user_indices2,user_indices3,user_indices4,user_indices5,\
#         user_indices6,user_indices7,user_indices8,user_indices9,user_indices10]
#    Uind.append(Uin)
#    Min=[movie_indices,movie_indices2,movie_indices3,movie_indices4,movie_indices5,\
#         movie_indices6,movie_indices7,movie_indices8,movie_indices9,movie_indices10]
#    Mind.append(Min)
#    L=[labels,labels2,labels3,labels4,labels5,labels6,labels7,labels8,labels9,labels10]
#    Label.append(L)
#    Tr=[train_mask,train_mask2,train_mask3,train_mask4,train_mask5,train_mask6,train_mask7,train_mask8,train_mask9,train_mask10]
#    Tramask.append(Tr)
#    Te=[test_mask,test_mask2,test_mask3,test_mask4,test_mask5,test_mask6,test_mask7,test_mask8,test_mask9,test_mask10]
#    Testmask.append(Te)
##
##保存数据
#import pickle
#output = open('LDNFSGBtenfolddata.pkl','wb')
#pickle.dump(U2madj,output)
#pickle.dump(M2uadj,output)
#pickle.dump(Uidf,output)
#pickle.dump(Midf,output)
#pickle.dump(Uind,output)
#pickle.dump(Mind,output)
#pickle.dump(Label,output)
#pickle.dump(Tramask,output)
#pickle.dump(Testmask,output)
#output.close()

