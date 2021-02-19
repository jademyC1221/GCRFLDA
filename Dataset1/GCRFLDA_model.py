#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 00:59:55 2021

@author: lsz
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class StackGCNEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_support,num_lnc,num_dis,
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
            self.bias_dis = nn.Parameter(torch.Tensor(output_dim, ))

        self.weight = init.kaiming_uniform_(self.weight) 
        
        self.weight_lnc = nn.Parameter(torch.Tensor(1, num_lnc, num_lnc))
        self.weight_dis = nn.Parameter(torch.Tensor(1, num_dis, num_dis))
        self.weight_lnc = init.kaiming_uniform_(self.weight_lnc)
        self.weight_dis = init.kaiming_uniform_(self.weight_dis)

    def crf_layer(self,hidden,hidden_new,flag):    
       #
       alpha = 50  
       beta = 1
       
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
                   #
       hidden_crf = torch.div(hidden_crf,const)        
       
       return hidden_crf

    def forward(self, lnc_supports, dis_supports, lnc_inputs, dis_inputs):

        assert len(lnc_supports) == len(dis_supports) == self.num_support

        lnc_hidden = []
        dis_hidden = []

        for i in range(self.num_support):

            tmp_u = torch.matmul(lnc_inputs, self.weight[i])
            tmp_v = torch.matmul(dis_inputs, self.weight[i])        

            tmp_lnc_hidden = torch.sparse.mm(lnc_supports[i], tmp_v)  
            tmp_dis_hidden = torch.sparse.mm(dis_supports[i], tmp_u) 
            
            hidden_lnc = tmp_lnc_hidden.cpu().detach().numpy()     
            hidden_new_lnc = tmp_lnc_hidden.cpu().detach().numpy() 
            hidden_dis = tmp_dis_hidden.cpu().detach().numpy()      
            hidden_new_dis = tmp_dis_hidden.cpu().detach().numpy()  
            
            #The CRF layer
            flag = 0
            lnc_hidden_crf = self.crf_layer(hidden_lnc,hidden_new_lnc,flag) 
            flag = 1
            dis_hidden_crf = self.crf_layer(hidden_dis,hidden_new_dis,flag)
            
#            flag = 0
#            for cv in range(0,1):  
#               lnc_hidden_crf = self.crf_layer(hidden_lnc,hidden_new_lnc,flag)  
#               hidden_new_lnc = lnc_hidden_crf
#            flag = 1
#            for cv in range(0,1):  
#               dis_hidden_crf = self.crf_layer(hidden_dis,hidden_new_dis,flag)  
#               hidden_new_dis = dis_hidden_crf
                        
            lnc_hidden.append(lnc_hidden_crf)
            dis_hidden.append(dis_hidden_crf)

        lnc_hidden = torch.cat(lnc_hidden, dim=1)  
        dis_hidden = torch.cat(dis_hidden, dim=1)  

        lnc_outputs = self.activation(lnc_hidden)  
        dis_outputs = self.activation(dis_hidden)  

        if self.use_bias:
            lnc_outputs += self.bias
            dis_outputs += self.bias_dis

        return lnc_outputs, dis_outputs


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
        self.linear_lnc = nn.Linear(input_dim, output_dim, bias=use_bias)      
        if self.share_weights:
            self.linear_dis = self.linear_lnc
        else:
            self.linear_dis = nn.Linear(input_dim, output_dim, bias=use_bias)  
        self.dropout = nn.Dropout(dropout)

    def forward(self, lnc_inputs, dis_inputs):

        lnc_inputs = self.dropout(lnc_inputs)
        lnc_outputs = self.linear_lnc(lnc_inputs)      
        
        dis_inputs = self.dropout(dis_inputs)
        dis_outputs = self.linear_dis(dis_inputs)     

        if self.activation:
            lnc_outputs = self.activation(lnc_outputs)  
            dis_outputs = self.activation(dis_outputs)  

        return lnc_outputs, dis_outputs  

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

    def forward(self, lnc_inputs, dis_inputs, lnc_indices, dis_indices):

        lnc_inputs = self.dropout(lnc_inputs) 
        dis_inputs = self.dropout(dis_inputs)  
        lnc_inputs = lnc_inputs[lnc_indices]  
        dis_inputs = dis_inputs[dis_indices]  
        
        basis_outputs = []
        for i in range(self.num_weights):
            tmp = torch.matmul(lnc_inputs, self.weight[i])        
            out = torch.sum(tmp * dis_inputs, dim=1, keepdim=True) 
            basis_outputs.append(out)

        basis_outputs = torch.cat(basis_outputs, dim=1)  
        
        outputs = torch.matmul(basis_outputs, self.weight_classifier)
        outputs = self.activation(outputs)
        
        return outputs