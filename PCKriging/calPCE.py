# Filename: calPCE.py

# Created: July/02/2018
# Last modified: July/07/2018
# Author: Prof. Leifur Leifsson
#         PhD student: Xiaosong        

'''PCE construction
   with quadrature method
   and collocation (OLS / LARS) method
'''

import math
import numpy as npy
import multiIters
import algPCE
import collections
from sklearn import linear_model
import sys

def quadrature(n_deg, x_prob, full_model):
    
    '''inputs:
       n_deg: required degree of PCE, scalar
       x_prob: random-variable information, dictionary
       full_model: real model for evaluation use, function
       outputs:
       PCE: key information of generated PCE model, dictionary
    '''
    
    # PCE basis, weighting factors, quadrature points, for 1D random variables
    weight= npy.zeros((n_deg+1, len(x_prob)))
    x_quad= npy.zeros((n_deg+1, len(x_prob)))
    x_quad_pred = npy.zeros((n_deg+1, len(x_prob)))
    basis = npy.zeros((n_deg+1, n_deg+1, len(x_prob)))
    integration = npy.zeros((n_deg+1, n_deg+1, len(x_prob)))
    
    # PCE coefficients, PCE basis, in multi-dimensional scale
    alpha_total = npy.zeros((math.factorial(n_deg+len(x_prob))/math.factorial(n_deg)/math.factorial(len(x_prob)),1))
    basis_total = npy.ones(((n_deg+1)**len(x_prob), math.factorial(n_deg+len(x_prob))/math.factorial(n_deg)/math.factorial(len(x_prob)), 1))
    integration_total = npy.ones(((n_deg+1)**len(x_prob), math.factorial(n_deg+len(x_prob))/math.factorial(n_deg)/math.factorial(len(x_prob)), 1))
    
    i = 0
    for key, value in x_prob.iteritems(): 
        # generate Gaussian quadrature weighting factors and points first
        [alpha, beta, x_quad[:, i], weight[:, i]] = algPCE.gen_quad(n_deg, key, value) 
        # use the obtained weighting factors, points, coefficients to generate basis
        [basis[:,:,i], integration[:,:,i]] = algPCE.gen_basis(n_deg, key, value, alpha, beta, x_quad[:, i], weight[:, i])
        # first save the x_quad for prediction use
        x_quad_pred[:, i] = x_quad[:, i]
        # then convert x_quad to the real range for real model
        x_quad[:, i] = algPCE.convert_x(x_quad[:, i], key, value)
        i = i + 1
    
    index = npy.array([])
    for iters in xrange(0, n_deg+1):
        index = npy.append(index, multiIters.multichoose(len(x_prob), iters))
    for iters in xrange(0, len(index)/len(x_prob)):
        index_cal = index[iters*len(x_prob):(iters+1)*len(x_prob)]
        basis_total[:,iters,0] = multiIters.iter_basis(basis, x_prob, n_deg, index_cal)
        integration_total[:,iters,0] = multiIters.iter_basis(integration, x_prob, n_deg, index_cal)
        
    weight_total = multiIters.iter_weights(weight, x_prob, n_deg)
    x_cal = multiIters.iter_x(x_quad, x_prob, n_deg)
    y_real = full_model(x_cal)
    
    alpha_total[:, 0] = npy.matmul(weight_total * y_real, basis_total[:,:,0])
    
    PCE = collections.OrderedDict([('PCE_coef', alpha_total),
                                   ('x_quad', x_quad_pred),
                                   ('weight_total', weight_total),
                                   ('mean', alpha_total[0]),
                                   ('variance', [npy.sum(alpha_total[1:]*alpha_total[1:])]),
                                   ('n_deg', n_deg),
                                   ('x_prob', x_prob),
                                   ('trunc_index', index)])
    
    return PCE
    
    
def collocation(n_deg, x_prob, x_exp, y_exp, meta_type):
    
    '''inputs:
       n_deg: required degree of PCE, scalar
       x_prob: random-input information, dictionary
       x_exp: sample points of random inputs
       y_exp: corresponding response of x_exp
       meta_type: OLS or LARS, string
       outputs:
       PCE: key information of generated PCE model, dictionary
    '''
    
    # PCE basis, weighting factors, quadrature points, for 1D random variables
    weight= npy.zeros((n_deg+1, len(x_prob)))
    x_quad= npy.zeros((n_deg+1, len(x_prob)))
    basis_coef = npy.zeros((len(x_exp), n_deg+1, len(x_prob)))
    
    # PCE coefficients, PCE basis, in multi-dimensional scale
    alpha_total = npy.zeros((math.factorial(n_deg+len(x_prob))/math.factorial(n_deg)/math.factorial(len(x_prob)),1))
    basis_coef_total = npy.ones((len(x_exp), math.factorial(n_deg+len(x_prob))/math.factorial(n_deg)/math.factorial(len(x_prob)), 1))
        
    i = 0
    for key, value in x_prob.iteritems(): 
        # generate Gaussian quadrature weighting factors and points first
        [alpha, beta, x_quad[:, i], weight[:, i]] = algPCE.gen_quad(n_deg, key, value) 
        # convert x_exp to the specified range for each polynomial basis
        x_exp[:, i] = algPCE.convert_x_inv(x_exp[:, i], key, value)
        [basis_coef[:,:,i], non] = algPCE.gen_basis(n_deg, key, value, alpha, beta, x_exp[:, i], weight[:, i])
        # convert x_exp back to the real range
        x_exp[:, i] = algPCE.convert_x(x_exp[:, i], key, value)
        i = i + 1
    
    index = npy.array([])
    for iters in xrange(0, n_deg+1):
        index = npy.append(index, multiIters.multichoose(len(x_prob), iters))
        
    for i in xrange(0, len(x_exp)):
        for iters in xrange(0, math.factorial(n_deg+len(x_prob))/math.factorial(n_deg)/math.factorial(len(x_prob))):
            for j in xrange(0, len(x_prob)):
                basis_coef_total[i,iters,0] = basis_coef_total[i,iters,0]*basis_coef[i,int(index[iters*len(x_prob)+j]),j]
    
    weight_total = multiIters.iter_weights(weight, x_prob, n_deg)
    
    if meta_type == 'OLS':
        reg = linear_model.LinearRegression()
        reg.__init__(fit_intercept=False, normalize=True, copy_X=True, n_jobs=1)
        
        reg.fit(basis_coef_total[:,:,0], y_exp)
            
        alpha_total[:, 0] = reg.coef_
        
        coef_index = [i for i, alpha_value in enumerate(alpha_total) if npy.abs(alpha_value) > 1e-4]
            
    elif meta_type == 'LARS':
            
        reg = linear_model.LassoLarsCV()
        reg.__init__(fit_intercept=True, verbose=False, normalize=True, copy_X=True, positive=False, cv=5)
        
        reg.fit(basis_coef_total[:,:,0], y_exp)
        alpha_total[:, 0] = reg.coef_
        alpha_total[0, 0] = reg.intercept_

        # this is actually the hybrid lars as mentioned in [Blatman and Sudret, 2011]
        # make the last step of LARS as OLS on selected basis (which have nonzero coefficients)
        coef_index = [i for i, alpha_value in enumerate(alpha_total) if npy.abs(alpha_value) > 1e-4]
        
        reg_ols = linear_model.LinearRegression()
        reg_ols.__init__(fit_intercept=False, normalize=True, copy_X=True, n_jobs=1)
        
        reg_ols.fit(basis_coef_total[:,coef_index,0], y_exp)
            
        alpha_total[coef_index, 0] = reg_ols.coef_

    else:
        print 'no such meta_type found'
        print 'now exiting!'
        sys.exit()    
    
    PCE = collections.OrderedDict([('PCE_coef', alpha_total),
                                   ('x_quad', x_quad),
                                   ('weight_total', weight_total),
                                   ('mean', alpha_total[0]),
                                   ('variance', [npy.sum(alpha_total[1:]*alpha_total[1:])]),
                                   ('n_deg', n_deg),
                                   ('x_prob', x_prob),
                                   ('trunc_index', index),
                                   ('basis_coef_total', basis_coef_total),
                                   ('coef_index', coef_index)])
    
    return PCE
    
