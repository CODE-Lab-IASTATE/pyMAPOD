# Filename: multiIters.py

# Created: July/02/2018
# Last modified: July/07/2018
# Author: Prof. Leifur Leifsson
#         PhD student: Xiaosong  

import numpy as npy

def multichoose(n,k):
    '''input arguments:
       n: dimension of random inputs
       k: kth order of PCE, note: not all orders provided
    '''
    if k < 0 or n < 0: return "Error"
    if not k: return [[0]*n]
    if not n: return []
    if n == 1: return [[k]]
    return [[val[0]+1]+val[1:] for val in multichoose(n,k-1)] + \
            [[0]+val for val in multichoose(n-1,k)]
            
def iter_basis(basis, x_prob, n_deg, index):
    
    basis_total = npy.array(([]))
    
    if len(basis[0,0,:]) == 1:
        return basis[:,int(index[0]),-1]
        
    for i in xrange(0, n_deg+1):
        basis_total = npy.append(basis_total, basis[i,int(index[0]),0]*iter_basis(basis[:,:,1:], x_prob, n_deg, index[1:]))
              
    return basis_total

def iter_weights(weight, x_prob, n_deg):

    weight_total = npy.array(([]))

    if len(weight[0,:]) == 1:
        return weight[:,-1]
        
    for i in xrange(0, n_deg+1):
        weight_total = npy.append(weight_total, weight[i,0]*iter_weights(weight[:,1:], x_prob, n_deg))
        
    return weight_total
    
def iter_x(x_quad, x_prob, n_deg):
    
    x_cal = npy.zeros([(n_deg+1)**len(x_prob), len(x_prob)])
    
    # id_int = npy.zeros([len(x_prob)])
    # id_rsd = npy.zeros([len(x_prob)])
    
    for i in xrange(0, (n_deg+1)**len(x_prob)):
        ids = i
        for ii in xrange(0, len(x_prob)):
            id_int = ids/(n_deg+1)
            id_rsd = ids%(n_deg+1)
            x_cal[i,len(x_prob)-1-ii] = x_quad[id_rsd, len(x_prob)-1-ii]
            ids = id_int
        
    return x_cal  