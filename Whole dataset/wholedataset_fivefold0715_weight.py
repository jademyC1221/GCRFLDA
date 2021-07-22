#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 01:13:40 2021

@author: lsz
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 22:28:36 2021

@author: lsz
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 02:24:00 2021

@author: lsz
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

    movie2user_adjacencies = globally_normalize_bipartite_adjacency(movie2user_adjacencies,
                                                                    symmetric=symmetric_normalization)

    return user2movie_adjacencies, movie2user_adjacencies


def get_node_identity_feature(num_user, num_movie):
    """one-hot encoding for nodes"""
    identity_feature = np.identity(num_user + num_movie, dtype=np.float32) 
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

    edge_df = pd.read_excel('./'+filename,sheet_name=sheetName,header=None)

    columns2 = ['lncnode','disnode','label','lncid','disid','usage']
    for i in range(6):
     edge_df.rename(columns={edge_df.columns[i]: columns2[i]},inplace=True)
    
    edge_df.loc[edge_df['usage']==2222,'usage'] = 'train'
    edge_df.loc[edge_df['usage']==1111,'usage'] = 'test'
    
    return edge_df

#    @staticmethod
def build_graph_adj(edge_df, symmetric_normalization=False):
    
    node_user = edge_df[['lncnode']].drop_duplicates().sort_values('lncnode')  #(240,1)
    node_movie = edge_df[['disnode']].drop_duplicates().sort_values('disnode') #(412,1)
    
    num_user = len(node_user)    
    num_movie = len(node_movie) 

    # adjacency
    user2movie_adjacencies, movie2user_adjacencies = get_adjacency(edge_df, num_user, num_movie,
                                                                   symmetric_normalization)

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
        """对得到的每类评分使用级联的方式进行聚合
        
        Args:
        ----
            input_dim (int): 输入的特征维度
            output_dim (int): 输出的特征维度，需要output_dim % num_support = 0
            num_support (int): 评分的类别数，比如1~5分，值为5
            use_bias (bool, optional): 是否使用偏置. Defaults to False.
            activation (optional): 激活函数. Defaults to F.relu.
        """
        super(StackGCNEncoder, self).__init__()
        self.input_dim = input_dim      #652
        self.output_dim = output_dim    #500
        self.num_support = num_support  #2
        self.use_bias = use_bias
        self.activation = activation
        assert output_dim % num_support == 0
        self.weight = nn.Parameter(torch.Tensor(num_support, 
            input_dim, output_dim // num_support))                     #torch.Size([2, 652, 250])
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim, ))
            self.bias_item = nn.Parameter(torch.Tensor(output_dim, ))
#        self.reset_parameters()
        self.weight = init.kaiming_uniform_(self.weight)  #torch.Size([2, 652, 250]) kaiming均匀分布
        
        self.weight_lnc = nn.Parameter(torch.Tensor(1, num_user, num_user))
        self.weight_dis = nn.Parameter(torch.Tensor(1, num_movie, num_movie))
        self.weight_lnc = init.kaiming_uniform_(self.weight_lnc)
        self.weight_dis = init.kaiming_uniform_(self.weight_dis)

    def crf_layer(self,hidden,hidden_new,flag):    
       #
       alpha = 50  
       beta = 1 #50
       
       hidden_extend = torch.from_numpy(hidden).float().unsqueeze(0)  #增加一个维度[1,240,250]
    
    #   attention
       conv1 = torch.nn.Conv1d(in_channels=250,out_channels=250,kernel_size=1) #卷积核大小=1*250
       hidden_extend = hidden_extend.permute(0,2,1)  #torch.Size([1, 250, 240])
       seq_fts = conv1(hidden_extend)                #torch.Size([1, 250, 240])
        
       conv1 = torch.nn.Conv1d(in_channels=250,out_channels=1,kernel_size=1)
       f_1 = conv1(seq_fts)
       conv1 = torch.nn.Conv1d(in_channels=250,out_channels=1,kernel_size=1)
       f_2 = conv1(seq_fts)
       logits = f_1 + f_2.permute(0, 2, 1)                 #(1, 240, 240)
        
       m = torch.nn.LeakyReLU(0.1)
       if flag==0:
           coefs = torch.nn.functional.softmax(m(logits)+self.weight_lnc)     
       elif flag==1:
           coefs = torch.nn.functional.softmax(m(logits)+self.weight_dis)     
       coefs = coefs[0] #(240,240)
       
       # fenzi
       coefs = coefs.float()                                  #(240,240)
       hidden_new = torch.from_numpy(hidden_new).float()      #(240,250)
       res = torch.mm(coefs, hidden_new)                      #[240, 250]
       hidden_neighbor = torch.mul(res,beta)                  #tf.multiply：乘法，相同位置的元素相乘
       hidden = torch.from_numpy(hidden).float()              #(240,250)
       hidden_self = torch.mul(hidden,alpha)                    
       hidden_crf = hidden_neighbor+hidden_self               #[240, 250]
       # fenmu
       unit_mat = torch.ones(hidden.shape[0],hidden.shape[1]).float()   #[240, 250]全1矩阵
       res = torch.mm(coefs, unit_mat)
       coff_sum = torch.mul(res,beta)
       const = coff_sum + torch.mul(unit_mat,alpha) 
                   #更新后的特征矩阵
       hidden_crf = torch.div(hidden_crf,const)        #矩阵对应位置的元素相除,torch.Size([240, 250])
       
       return hidden_crf

    def forward(self, user_supports, item_supports, user_inputs, item_inputs):
        """StackGCNEncoder计算逻辑
        Args:
            user_supports (list of torch.sparse.FloatTensor): 
                归一化后每个评分等级对应的用户与商品邻接矩阵
            item_supports (list of torch.sparse.FloatTensor):
                归一化后每个评分等级对应的商品与用户邻接矩阵
            user_inputs (torch.Tensor): 用户特征的输入
            item_inputs (torch.Tensor): 商品特征的输入
        
        Returns:
            [torch.Tensor]: 用户的隐层特征
            [torch.Tensor]: 商品的隐层特征
        """
        assert len(user_supports) == len(item_supports) == self.num_support
        # user_supports = user2movie_adjacencies[0]
        # item_supports = movie2user_adjacencies[0]
        # user_inputs=user_side_feature, item_inputs=movie_side_feature(×)
        # user_inputs=user_identity_feature, item_inputs=movie_identity_feature(√)
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

        user_hidden = torch.cat(user_hidden, dim=1)  #240x500
        item_hidden = torch.cat(item_hidden, dim=1)  #412x500

        user_outputs = self.activation(user_hidden)  #[240, 500]
        item_outputs = self.activation(item_hidden)  #[412, 500]

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
        
        self.dropout = nn.Dropout(dropout)  

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        init.kaiming_uniform_(self.weight_classifier)

    def forward(self, user_inputs, item_inputs, user_indices, item_indices):

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
#    
#
#
# In[]

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn import metrics
from scipy import interp
from sklearn.metrics import roc_curve, auc


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
        self.encoder = StackGCNEncoder(input_dim, gcn_hidden_dim, num_support,num_user,num_movie)
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
    
##
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
#
#
#filename='wholeset_edge_f5cv2.xlsx'
#edge_df = read_edge2(filename=filename,sheetName='Sheet 1')
#user2movie_adjacencies, movie2user_adjacencies, \
#    user_identity_feature, movie_indentity_feature, \
#    user_indices, movie_indices, labels, train_mask, test_mask = build_graph_adj(edge_df=edge_df, symmetric_normalization=False)
#
#edge_df2 = read_edge2(filename=filename,sheetName='Sheet 2')
#user2movie_adjacencies2, movie2user_adjacencies2, \
#    user_identity_feature2, movie_indentity_feature2, \
#    user_indices2, movie_indices2, labels2, train_mask2, test_mask2 = build_graph_adj(edge_df=edge_df2, symmetric_normalization=False)
#
#edge_df3 = read_edge2(filename=filename,sheetName='Sheet 3')
#user2movie_adjacencies3, movie2user_adjacencies3, \
#    user_identity_feature3, movie_indentity_feature3, \
#    user_indices3, movie_indices3, labels3, train_mask3, test_mask3 = build_graph_adj(edge_df=edge_df3, symmetric_normalization=False)
#
#edge_df4 = read_edge2(filename=filename,sheetName='Sheet 4')
#user2movie_adjacencies4, movie2user_adjacencies4, \
#    user_identity_feature4, movie_indentity_feature4, \
#    user_indices4, movie_indices4, labels4, train_mask4, test_mask4 = build_graph_adj(edge_df=edge_df4, symmetric_normalization=False)
#
#edge_df5 = read_edge2(filename=filename,sheetName='Sheet 5')
#user2movie_adjacencies5, movie2user_adjacencies5, \
#    user_identity_feature5, movie_indentity_feature5, \
#    user_indices5, movie_indices5, labels5, train_mask5, test_mask5 = build_graph_adj(edge_df=edge_df5, symmetric_normalization=False)
#
#
#U2madj=[user2movie_adjacencies,user2movie_adjacencies2,user2movie_adjacencies3,user2movie_adjacencies4,user2movie_adjacencies5]
#
#M2uadj=[movie2user_adjacencies,movie2user_adjacencies2,movie2user_adjacencies3,movie2user_adjacencies4,movie2user_adjacencies5]
#
#Uidf=[user_identity_feature,user_identity_feature2,user_identity_feature3,user_identity_feature4,user_identity_feature5]
#
#Midf=[movie_indentity_feature,movie_indentity_feature2,movie_indentity_feature3,movie_indentity_feature4,movie_indentity_feature5]
#
#Uind=[user_indices,user_indices2,user_indices3,user_indices4,user_indices5]
#
#Mind=[movie_indices,movie_indices2,movie_indices3,movie_indices4,movie_indices5]
#
#Label=[labels,labels2,labels3,labels4,labels5]
#
#Tramask=[train_mask,train_mask2,train_mask3,train_mask4,train_mask5]
#
#Testmask=[test_mask,test_mask2,test_mask3,test_mask4,test_mask5]
#
#
##保存数据
#import pickle
#output = open('wholeset_fivefold_data.pkl','wb')
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

#
import pickle
pkl_file=open('wholeset_fivefold_data.pkl','rb')
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
#
user2movie_adjacencies = U2madj[0]
user2movie_adjacencies2 = U2madj[1]
user2movie_adjacencies3 = U2madj[2]
user2movie_adjacencies4 = U2madj[3]
user2movie_adjacencies5 = U2madj[4]

movie2user_adjacencies = M2uadj[0]
movie2user_adjacencies2 = M2uadj[1]
movie2user_adjacencies3 = M2uadj[2]
movie2user_adjacencies4 = M2uadj[3]
movie2user_adjacencies5 = M2uadj[4]

user_identity_feature = Uidf[0]
user_identity_feature2 = Uidf[1]
user_identity_feature3 = Uidf[2]
user_identity_feature4 = Uidf[3]
user_identity_feature5 = Uidf[4]

movie_indentity_feature = Midf[0]
movie_indentity_feature2 = Midf[1]
movie_indentity_feature3 = Midf[2]
movie_indentity_feature4 = Midf[3]
movie_indentity_feature5 = Midf[4]

user_indices=Uind[0]
user_indices2=Uind[1]
user_indices3=Uind[2]
user_indices4=Uind[3]
user_indices5=Uind[4]

movie_indices=Mind[0]
movie_indices2=Mind[1]
movie_indices3=Mind[2]
movie_indices4=Mind[3]
movie_indices5=Mind[4]

labels=Label[0]
labels2=Label[1]
labels3=Label[2]
labels4=Label[3]
labels5=Label[4]

train_mask=Tramask[0]
train_mask2=Tramask[1]
train_mask3=Tramask[2]
train_mask4=Tramask[3]
train_mask5=Tramask[4]

test_mask=Testmask[0]
test_mask2=Testmask[1]
test_mask3=Testmask[2]
test_mask4=Testmask[3]
test_mask5=Testmask[4]
#
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

#
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


num_user = len(user_identity_feature)    #lnc
num_movie = len(movie_identity_feature)  #dis

from scipy.io import loadmat
datadict = loadmat("E2.mat")  
weightslist = datadict.get('E2') 

NUM_BASIS = 2
WEIGHT_DACAY = 0.
NODE_INPUT_DIM = user_identity_feature.shape[1]  #节点总数量

EPOCHS = 240      
DROPOUT_RATIO = 0.5    #0.25  0.5
LEARNING_RATE = 0.001   #学习率1e-2=0.01,0.001

LEARNING_RATE = 0.001   #学习率1e-2=0.01,0.001
GCN_HIDDEN_DIM = 500    #GCN隐藏层
SIDE_HIDDEN_DIM = 10    #边隐藏层20
ENCODE_HIDDEN_DIM = 50  #编码隐藏层 75
mean_fpr = np.linspace(0, 1, 100)
AUCresult=[]
AUPRresult=[]
CI=[]
PKL=[]

  
Weight=[]

#
iii=1
lncfilename="wholeset_lncsim28"
for jjj in range(6,9,1):
    disfilename="wholeset_dissim"+weightslist[jjj]
    lncfile="./Foursetsimlkf/" +lncfilename
    disfile="./Foursetsimlkf/" +disfilename
    
    users_df, movie_df = read_data(lncfile,disfile,lncfilename,disfilename)
    SIDE_FEATURE_DIM = users_df.shape[1]+movie_df.shape[1]
    
    user_side_feature, movie_side_feature = build_graph_df(lnclnc=users_df, disdis=movie_df, symmetric_normalization=False)
    user_side_feature      = tensor_from_numpy(user_side_feature).float()
    movie_side_feature     = tensor_from_numpy(movie_side_feature).float()
        
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
        

    tprs = []
    prs=[]
    aucs = []
    auprcs=[]
    testprob =  []
    testpred_y = []
    y_real = []
    losstest_history=[]
    losstrian_history=[]
    Epochnum=[]
    
    for i in range(5):
    
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
    
        else :
            model_inputs = model_inputs5
            train_mask = train_mask5
            test_mask = test_mask5
            labels = labels5
        
        model.train()
        for e in range(EPOCHS):
    
            logits = model(*model_inputs)
            loss = criterion(logits[train_mask], labels[train_mask])
    
            optimizer.zero_grad()
            loss.backward()   # 反向传播计算参数的梯度
            optimizer.step()  # 使用优化方法进行梯度更新
            
            lossitem=loss.item()
            losstrian_history.append(lossitem)
    
            train_acc = accu(train_mask)
    #            losstrain_history = losstrain_history+[loss.item()]
            print("Epoch {:03d}: Loss: {:.4f}, TrainAcc {:.4}".format(e, loss.item(), train_acc.item()))            
    #            losstrain_5fold.append(losstrain_history)
    
        
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
    #        prs.append(interp(mean_fpr, fpr, tpr))
            y_real.append(true_y)
            auprc = auc(recall, precision)
            auprcs.append(auprc)
    
            test_acc = accu(test_mask)
            lossitem=loss.item()
            test_accitem = test_acc.item()
#                    print('test on....................iii=%d' % (iii))
    
            print('Test On Epoch {}: loss: {:.4f}, Testauc {:.4}, Testaupr {:.4},'.format(e, loss.item(), roc_auc, auprc))
            Epochnum.append(e)
            losstest_history.append(lossitem)
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
    
    AUCresult.append(mean_auc)
    AUPRresult.append(mean_aupr)
    weight=[weightslist[iii],weightslist[jjj]]
    Weight.append(weight)
    from scipy import stats
    ci = stats.t.interval(alpha=0.95, df=len(aucs) - 1, loc=np.mean(aucs), scale=stats.sem(aucs))
    CI.append(ci)
    
    
    namepkl = 'Weight_wholeset4339_epoch240drop0.5lr0.001'+weightslist[iii]+weightslist[jjj]+'.pkl'
    PKL.append(namepkl)
                
    #保存数据
#                import pickle
    output = open(namepkl,'wb')
    pickle.dump(aucs,output)
    pickle.dump(auprcs,output)
    pickle.dump(Epochnum,output)
    pickle.dump(losstest_history,output)
    pickle.dump(losstrian_history,output)
    pickle.dump(AUCresult,output)
    pickle.dump(AUPRresult,output)
    pickle.dump(CI,output)
    pickle.dump(PKL,output)
    pickle.dump(Weight,output)
    pickle.dump(tprs,output)
    pickle.dump(mean_auc,output)
    pickle.dump(std_auc,output)
    pickle.dump(std_tpr,output)
    pickle.dump(y_real,output)
    pickle.dump(y_proba,output)
    pickle.dump(mean_aupr,output)
    output.close()

#
for iii in range(2,9,1):   
    lncfilename="wholeset_lncsim"+weightslist[iii]
    for jjj in range(0,9,1):
        disfilename="wholeset_dissim"+weightslist[jjj]
        lncfile="./Foursetsimlkf/" +lncfilename
        disfile="./Foursetsimlkf/" +disfilename
        
        users_df, movie_df = read_data(lncfile,disfile,lncfilename,disfilename)
        SIDE_FEATURE_DIM = users_df.shape[1]+movie_df.shape[1]
        
        user_side_feature, movie_side_feature = build_graph_df(lnclnc=users_df, disdis=movie_df, symmetric_normalization=False)
        user_side_feature      = tensor_from_numpy(user_side_feature).float()
        movie_side_feature     = tensor_from_numpy(movie_side_feature).float()
            
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
            

        tprs = []
        prs=[]
        aucs = []
        auprcs=[]
        testprob =  []
        testpred_y = []
        y_real = []
        losstest_history=[]
        losstrian_history=[]
        Epochnum=[]
        
        for i in range(5):
        
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
        
            else :
                model_inputs = model_inputs5
                train_mask = train_mask5
                test_mask = test_mask5
                labels = labels5
            
            model.train()
            for e in range(EPOCHS):
        
                logits = model(*model_inputs)
                loss = criterion(logits[train_mask], labels[train_mask])
        
                optimizer.zero_grad()
                loss.backward()   # 反向传播计算参数的梯度
                optimizer.step()  # 使用优化方法进行梯度更新
                
                lossitem=loss.item()
                losstrian_history.append(lossitem)
        
                train_acc = accu(train_mask)
        #            losstrain_history = losstrain_history+[loss.item()]
                print("Epoch {:03d}: Loss: {:.4f}, TrainAcc {:.4}".format(e, loss.item(), train_acc.item()))            
        #            losstrain_5fold.append(losstrain_history)
        
            
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
        #        prs.append(interp(mean_fpr, fpr, tpr))
                y_real.append(true_y)
                auprc = auc(recall, precision)
                auprcs.append(auprc)
        
                test_acc = accu(test_mask)
                lossitem=loss.item()
                test_accitem = test_acc.item()
#                    print('test on....................iii=%d' % (iii))
        
                print('Test On Epoch {}: loss: {:.4f}, Testauc {:.4}, Testaupr {:.4},'.format(e, loss.item(), roc_auc, auprc))
                Epochnum.append(e)
                losstest_history.append(lossitem)
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
        
        AUCresult.append(mean_auc)
        AUPRresult.append(mean_aupr)
        weight=[weightslist[iii],weightslist[jjj]]
        Weight.append(weight)
#        from scipy import stats
        ci = stats.t.interval(alpha=0.95, df=len(aucs) - 1, loc=np.mean(aucs), scale=stats.sem(aucs))
        CI.append(ci)
        
        
        namepkl = 'Weight_wholeset4339_epoch240drop0.5lr0.001'+weightslist[iii]+weightslist[jjj]+'.pkl'
        PKL.append(namepkl)
                    
        #保存数据
    #                import pickle
        output = open(namepkl,'wb')
        pickle.dump(aucs,output)
        pickle.dump(auprcs,output)
        pickle.dump(Epochnum,output)
        pickle.dump(losstest_history,output)
        pickle.dump(losstrian_history,output)
        pickle.dump(AUCresult,output)
        pickle.dump(AUPRresult,output)
        pickle.dump(CI,output)
        pickle.dump(PKL,output)
        pickle.dump(Weight,output)
        pickle.dump(tprs,output)
        pickle.dump(mean_auc,output)
        pickle.dump(std_auc,output)
        pickle.dump(std_tpr,output)
        pickle.dump(y_real,output)
        pickle.dump(y_proba,output)
        pickle.dump(mean_aupr,output)
        output.close()
        
# #读取数据
#import pickle
#pkl_file=open('Weight_wholeset4339_epoch240drop0.5lr0.001'+weightslist[iii]+weightslist[jjj]+'.pkl','rb')
#aucs=pickle.load(pkl_file)
#auprcs=pickle.load(pkl_file)
#Epochnum=pickle.load(pkl_file)
#losstest_history=pickle.load(pkl_file)
#losstrian_history=pickle.load(pkl_file)
#AUCresult=pickle.load(pkl_file)
#AUPRresult=pickle.load(pkl_file)
#CI=pickle.load(pkl_file)
#PKL=pickle.load(pkl_file)
#Weight=pickle.load(pkl_file)
#tprs =pickle.load(pkl_file)
#mean_auc =pickle.load(pkl_file)
#std_auc =pickle.load(pkl_file)
#std_tpr =pickle.load(pkl_file)
#y_real =pickle.load(pkl_file)
#y_proba =pickle.load(pkl_file)
#mean_aupr =pickle.load(pkl_file)
#pkl_file.close()