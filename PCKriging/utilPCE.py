# Filename: utilPCE.py

# Created: July/08/2018
# Last modified: July/08/2018
# Author: Prof. Leifur Leifsson
#         PhD student: Xiaosong        

import numpy as npy
import math
import algPCE
import multiIters
import collections

'''PCE predictor
   using obtained information
   only need to reconstruct basis for prediction points
'''

def predictor(PCE, x_pred):
     
    '''inputs:
       PCE: PCE model, dictionary
       x_pred: random inputs to be predicted for
       outputs:
       y_pred: predition on model response, array
    '''
    
    # define array of random inputs on the definition range of polynomial basis
    x_prediction = npy.zeros(x_pred.shape)
    # PCE basis for 1D random variables
    basis = npy.zeros((len(x_pred[:,0]), PCE['n_deg']+1, len(PCE['x_prob'])))
    # PCE basis for mD random variables
    basis_total = npy.ones((len(x_pred), math.factorial(PCE['n_deg']+len(PCE['x_prob']))/math.factorial(PCE['n_deg'])/math.factorial(len(PCE['x_prob']))))
    
    i = 0
    for key, value in PCE['x_prob'].iteritems():
        # generate Gaussian quadrature weighting factors and points first
        [alpha, beta, x_quad, weight] = algPCE.gen_quad(PCE['n_deg'], key, value) 
        # convert x_pred to the definition range of polynomial basis
        x_prediction[:, i] = algPCE.convert_x_inv(x_pred[:, i], key, value)
        # use the obtained weighting factors, points, coefficients to generate basis
        [basis[:,:,i], non] = algPCE.gen_basis(PCE['n_deg'], key, value, alpha, beta, x_prediction[:, i], weight)
        i = i + 1
        
    for i in xrange(0, len(x_pred[:,0])):
        for iters in xrange(0, math.factorial(PCE['n_deg']+len(PCE['x_prob']))/math.factorial(PCE['n_deg'])/math.factorial(len(PCE['x_prob']))):
            for j in xrange(0, len(PCE['x_prob'])):
                basis_total[i,iters] = basis_total[i,iters]*basis[i,int(PCE['trunc_index'][iters*len(PCE['x_prob'])+j]),j]
                
    y_pred = (PCE['PCE_coef'] * npy.transpose(basis_total)).sum(0)

    PCE_pred = collections.OrderedDict([('y_pred', y_pred),
                                        ('basis_total', basis_total)])
    
    return PCE_pred, y_pred
    

def validation(y_real, y_pred):
     
    '''inputs:
       y_real: real model response, array
       y_pred: predition on model response, array
       valType: rmse / nrmse
       outputs:
       RMSE: root mean squared error, scalar
       NRMSE: normalized root mean squared error, scalar
    '''
    
    RMSE = npy.sqrt(npy.sum((y_real - y_pred)*(y_real - y_pred)) / len(y_real))
    
    NRMSE = RMSE / (max(y_real) - min(y_real))
    
    return RMSE, NRMSE
