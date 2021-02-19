#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 00:58:36 2021

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


def get_adjacency(edge_df, num_lnc, num_dis, symmetric_normalization):
    
    lnc2dis_adjacencies = []
    dis2lnc_adjacencies = []
    train_edge_df = edge_df.loc[edge_df['usage'] == 'train']
    for i in range(2):
        edge_index = train_edge_df.loc[train_edge_df.label == i, ['lncid', 'disid']].to_numpy()
        support = sp.csr_matrix((np.ones(len(edge_index)), (edge_index[:, 0], edge_index[:, 1])),
                                shape=(num_lnc, num_dis), dtype=np.float32)
        lnc2dis_adjacencies.append(support)
        dis2lnc_adjacencies.append(support.T)

    lnc2dis_adjacencies = globally_normalize_bipartite_adjacency(lnc2dis_adjacencies,
                                                                    symmetric=symmetric_normalization)

    dis2lnc_adjacencies = globally_normalize_bipartite_adjacency(dis2lnc_adjacencies,
                                                                    symmetric=symmetric_normalization)

    return lnc2dis_adjacencies, dis2lnc_adjacencies


def get_node_identity_feature(num_lnc, num_dis):
    """one-hot encoding for nodes"""
    
    identity_feature = np.identity(num_lnc + num_dis, dtype=np.float32) 
    lnc_identity_feature, dis_indentity_feature = identity_feature[:num_lnc], identity_feature[num_lnc:]

    return lnc_identity_feature, dis_indentity_feature

def convert_to_homogeneous(lnc_feature: np.ndarray, dis_feature: np.ndarray):

    num_lnc, lnc_feature_dim = lnc_feature.shape
    num_dis, dis_feature_dim = dis_feature.shape
    lnc_feature = np.concatenate(
        [lnc_feature, np.zeros((num_lnc, dis_feature_dim))], axis=1)
    dis_feature = np.concatenate(
        [dis_feature, np.zeros((num_dis, lnc_feature_dim))], axis=1)

    return lnc_feature, dis_feature


def read_data(lncfile,disfile,lncfilename,disfilename):

    # lncfeature
    from scipy.io import loadmat
    lncfeadict = loadmat(lncfile+".mat")
    lncfeature = lncfeadict.get(lncfilename)
    # disfeature
    from scipy.io import loadmat
    disfeadict = loadmat(disfile+".mat")
    disfeature = disfeadict.get(disfilename)
    
    return lncfeature, disfeature

#def read_edge(filename,sheetName):
#
#    edge_df = pd.read_excel(filename,sheet_name=sheetName,header=0)
#    
#    return edge_df

def read_edge2(filename,sheetName):

    edge_df = pd.read_excel('./'+filename,sheet_name=sheetName,header=None)

    columns2 = ['lncnode','disnode','label','lncid','disid','usage']
    for i in range(6):
     edge_df.rename(columns={edge_df.columns[i]: columns2[i]},inplace=True)
    
    edge_df.loc[edge_df['usage']==2222,'usage'] = 'train'
    edge_df.loc[edge_df['usage']==1111,'usage'] = 'test'
    
    return edge_df


def build_graph_adj(edge_df, symmetric_normalization=False):
    
    node_lnc = edge_df[['lncnode']].drop_duplicates().sort_values('lncnode')  #(240,1)
    node_movie = edge_df[['disnode']].drop_duplicates().sort_values('disnode') #(412,1)
    
    num_lnc = len(node_lnc)    
    num_dis = len(node_movie) 

    # adjacency
    lnc2dis_adjacencies, dis2lnc_adjacencies = get_adjacency(edge_df, num_lnc, num_dis,
                                                                   symmetric_normalization)

    # one-hot encoding for nodes
    lnc_identity_feature, dis_indentity_feature = get_node_identity_feature(num_lnc, num_dis)

    # lnc_indices, dis_indices, labels, train_mask
    lnc_indices, dis_indices, labels = edge_df[['lncid', 'disid', 'label']].to_numpy().T
    train_mask = (edge_df['usage'] == 'train').to_numpy()
    test_mask = (edge_df['usage'] == 'test').to_numpy()

    return lnc2dis_adjacencies, dis2lnc_adjacencies, \
        lnc_identity_feature, dis_indentity_feature, \
        lnc_indices, dis_indices, labels, train_mask, test_mask



def build_graph_df(lnclnc, disdis, symmetric_normalization=False):

    # side information
    lnc_side_feature, dis_side_feature = convert_to_homogeneous(lnclnc,disdis)

    return lnc_side_feature, dis_side_feature