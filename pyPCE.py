# Filename: pyPCE
# Main function

# Created: July/02/2018
# Last modified: July/02/2018
# Author: Prof. Leifur Leifsson
#         PhD student: Xiaosong Du

#import probDistrs
from PCKriging import calPCE
import collections
#import funcDef
from scipy.io import loadmat
from PCKriging import utilPCE
import numpy as npy
import sys


def meta_gen(n_deg, x_pred, x_exp, data, x_prob, a, meta_type):

        if len(a) == 0:
            x_experiment = x_exp
            y_experiment = data[:, 2]
            PCE = calPCE.collocation(n_deg, x_prob, x_experiment, y_experiment, meta_type)
            PCE_pred, y_pred = utilPCE.predictor(PCE, x_pred)
            y_pred = y_pred.reshape(len(x_pred),1)
            y_pred[y_pred<0.0001] = min(y_pred[y_pred>0.0001])
            index = npy.arange(1, len(x_pred)+1)
            index.shape = (len(x_pred),1)
            x_a = x_pred[:,0].reshape(len(x_pred),1)
            data_meta = npy.concatenate((index, x_a, y_pred), axis = 1)
        else:
            data_meta = npy.zeros((len(a)*len(x_pred), 3))
            for i in xrange(0, len(a)):
                x_experiment = x_exp
                print(x_experiment.shape)
                y_experiment = data[i*len(x_exp):(i+1)*len(x_exp), 2]
                print(y_experiment.shape)
                PCE = calPCE.collocation(n_deg, x_prob, x_experiment, y_experiment, meta_type)
                PCE_pred, y_pred = utilPCE.predictor(PCE, x_pred)
                y_pred = y_pred.reshape(len(x_pred),1)
                y_pred[y_pred<0.0001] = min(y_pred[y_pred>0.0001])
                index = npy.arange(1+i*len(x_pred), (i+1)*len(x_pred)+1)
                index.shape = (len(x_pred),1)
                data_meta[i*len(x_pred):(i+1)*len(x_pred), :] = npy.concatenate((index, a[i]*npy.ones((len(x_pred),1)), y_pred), axis = 1)
        
        data = data_meta[data_meta[:,2] > min(data_meta[:,2])]
        return data

def meta_val(n_deg, x_pred, y_real, x_exp, data, x_prob, a, meta_type):

        if len(a) == 0:
            x_experiment = x_exp
            y_experiment = data[:, 2]
            PCE = calPCE.collocation(n_deg, x_prob, x_experiment, y_experiment, meta_type)
            PCE_pred, y_pred = utilPCE.predictor(PCE, x_pred)
            y_Real = y_real[:,2]
            rmse, nrmse = utilPCE.validation(y_Real, y_pred)
        else:
           rmse = npy.zeros(len(a))
           nrmse = npy.zeros(len(a)) 
           data_meta = npy.zeros((len(a)*len(x_pred), 3))
           for i in xrange(0, len(a)):
               x_experiment = x_exp
               y_experiment = data[i*len(x_exp):(i+1)*len(x_exp), 2]
               PCE = calPCE.collocation(n_deg, x_prob, x_experiment, y_experiment, meta_type)
               PCE_pred, y_pred = utilPCE.predictor(PCE, x_pred)
               y_Real = y_real[i*len(x_pred):(i+1)*len(x_pred), 2]
               rmse[i], nrmse[i] = utilPCE.validation(y_Real, y_pred)

        return rmse, nrmse


def sobol_gen(a, x_prob, dim, n_exp, x_mix, *args):
    x_exp1 = args[0]
    x_exp2 = args[1] 
    n_deg = args[2]
    x_exp = args[3]
    data = args[4]
    meta_type = args[5]
    y  = npy.zeros((n_exp, dim))
    y1 = npy.zeros((n_exp, 1))
    y2 = npy.zeros((n_exp, 1))
    x_experiment = x_exp
    y_experiment = data[:, 2]
    PCE = calPCE.collocation(n_deg, x_prob, x_experiment, y_experiment, meta_type)
    PCE_1, y1 = utilPCE.predictor(PCE, x_exp1)
    PCE_2, y2 = utilPCE.predictor(PCE, x_exp2)
    for i in xrange(0, dim):
        x_mix[:,:] = x_exp1
        x_mix[:,i] = x_exp2[:,i]
        PCE_dummy, y[:,i] = utilPCE.predictor(PCE, x_mix)

    return y1, y2, y
# define the real model
#full_model = funcDef.multi

# define random inputs, with specific statistical distributions
#x_prob = collections.OrderedDict([('Uniform', (2, 3)),
#                                  ('Gaussian', (3, 2)),
#                                  ('Gamma', (3, 2)),
#                                  ('Beta', (3, 4)),
#                                  ('Exponential', (4)),
#                                 ('Weibull', (2, 3)),
#                                  ('Laplace', (3, 4))])

# available sample methods: 'LHS', 'FF', 'FF2', 'FracF', 'PB', 'BB', 'CC'
# value is used as the NO. of sample points   
# assign sampling points (training points)
#x_sample = {'LHS' : 36}                                     
#x_exp = probDistrs.expGen(x_sample, x_prob)
#y_exp = full_model(x_exp) (data in main script)

# assign testing points
#x_test = {'MCS' : 1000}
#x_pred = probDistrs.expGen(x_test, x_prob)
#y_real = full_model(x_pred)

# define required degree of PCE
#n_deg = 2

# call quadrature method, only n_deg, x_prob and real model required
# actually this function cannot handle arbitrary (x,y) data
# PCE_quad includes: PCE_coef, x_quad, weight, alpha, beta, mean, variance
#PCE_quad = calPCE.quadrature(n_deg, x_prob, full_model)
#print PCE_quad['mean'], PCE_quad['variance']

# call collocation method, need n_deg, x_prob, and sample points
# both OLS and LARS are included here
# PCE_quad includes: PCE_coef, x_quad, weight, alpha, beta, mean, variance

# y = loadmat('multi_func.mat')
# x_exp = y['DVs']
# y_exp = y['response']

#PCE_ols = calPCE.collocation(n_deg, x_prob, x_exp, y_exp, 'OLS')
#print PCE_ols['mean'], PCE_ols['variance']

#PCE_lars = calPCE.collocation(n_deg, x_prob, x_exp, y_exp, 'LARS')
#print PCE_lars['mean'], PCE_lars['variance']

# y = loadmat('rmse_validation.mat')
# x_pred = y['DVs_test']
# y_real = full_model(x_pred)

# utilize the generated PCE for prediction use
#y_pred_quad = utilPCE.predictor(PCE_quad, x_pred)
#y_pred_ols  = utilPCE.predictor(PCE_ols,  x_pred)
#y_pred_lars = utilPCE.predictor(PCE_lars, x_pred)
# print y_real
# print y_pred_quad
# print y_pred_coll

# RMSE validation
#rmse_quad, nrmse_quad = utilPCE.validation(y_real, y_pred_quad)
#rmse_ols, nrmse_ols = utilPCE.validation(y_real, y_pred_ols)
#rmse_lars, nrmse_lars = utilPCE.validation(y_real, y_pred_lars)
#print rmse_quad, nrmse_quad
#print rmse_ols, nrmse_ols
#print rmse_lars, nrmse_lars
