# pyPCKriging

# basic settings for constructing Kriging metamodel
# call createKriging for the general process:
# 1. experimental design (generated by user)
# 2. trend function ("ordinary" by default)
# 3. correlation function ("anisotropic ellipsoidal, matern-5_2" by default)
# 4. estimation method ("cross validation" by default)
# 5. optimization method ("genetic algorithm" by default)

# created on October 18, 2018
# modified on October 18, 2018
# developed by Leifur Leifsson (assisted professor)
#              Xiaosong Du (Ph.D. student)

import numpy as npy
from PCKriging import createKriging
from scipy import io
import matplotlib.pyplot as plt


from PCKriging import calPCE
import collections
from PCKriging import utilPCE


def meta_gen(n_deg, x_pred, x_exp, data, x_prob, a, setting, lower_bound, upper_bound):

    if len(a) == 0:
        setting['exp_x'] = x_exp
        setting['exp_y'] = data[:,2]
        y_exp = data[:,2]
        PCE_lars = calPCE.collocation(n_deg, x_prob, x_exp, y_exp, 'LARS')
        setting['bounds'] = [(lower_bound, upper_bound), (lower_bound, upper_bound), (lower_bound, upper_bound)]
        model = createKriging.createKriging(setting, PCE_lars)
        PCE_pred, y_PCE_pred = utilPCE.predictor(PCE_lars, x_pred)
        y_kriging, ci_kriging, var_kriging = createKriging.predictor(x_pred, model, setting, PCE_lars, PCE_pred)
        y_pred = y_kriging.reshape(len(x_pred),1)
        y_pred[y_pred<0.0001] = min(y_pred[y_pred>0.0001])
        index = npy.arange(1, len(x_pred)+1)
        index.shape = (len(x_pred),1)
        x_a = x_pred[:,0].reshape(len(x_pred),1)
        data_meta = npy.concatenate((index, x_a, y_pred), axis = 1)
    else:
        data_meta = npy.zeros((len(a)*len(x_pred), 3))
        for i in xrange(0, len(a)):
            setting['exp_x'] = x_exp
            setting['exp_y'] = data[i*len(x_exp):(i+1)*len(x_exp), 2]
            y_exp = data[i*len(x_exp):(i+1)*len(x_exp), 2]
            PCE_lars = calPCE.collocation(n_deg, x_prob, x_exp, y_exp, 'LARS')
            setting['bounds'] = [(lower_bound, upper_bound), (lower_bound, upper_bound)]
            model = createKriging.createKriging(setting, PCE_lars)
            PCE_pred, y_PCE_pred = utilPCE.predictor(PCE_lars, x_pred)
            y_kriging, ci_kriging, var_kriging = createKriging.predictor(x_pred, model, setting, PCE_lars, PCE_pred)
            y_pred = y_kriging.reshape(len(x_pred),1)
            y_pred[y_pred<0.0001] = min(y_pred[y_pred>0.0001])
            index = npy.arange(1, len(x_pred)+1)
            index.shape = (len(x_pred),1)
            data_meta[i*len(x_pred):(i+1)*len(x_pred), :] = npy.concatenate((index, a[i]*npy.ones((len(x_pred),1)), y_pred), axis = 1)
   
    data = data_meta[data_meta[:,2] > min(data_meta[:,2])]
    return data 

def meta_val(n_deg, x_pred, y_real, x_exp, data, x_prob, a, setting, lower_bound, upper_bound):

        if len(a) == 0:
            setting['exp_x'] = x_exp
            setting['exp_y'] = data[:,2]
            y_exp = data[:,2]
            PCE_lars = calPCE.collocation(n_deg, x_prob, x_exp, y_exp, 'LARS')
            setting['bounds'] = [(lower_bound, upper_bound), (lower_bound, upper_bound), (lower_bound, upper_bound)]
            model = createKriging.createKriging(setting, PCE_lars)
            PCE_pred, y_PCE_pred = utilPCE.predictor(PCE_lars, x_pred)
            y_kriging, ci_kriging, var_kriging = createKriging.predictor(x_pred, model, setting, PCE_lars, PCE_pred)
            y_pred = y_kriging
            y_Real = y_real[:,2]
            rmse, nrmse = utilPCE.validation(y_Real, y_pred)
        else:
            rmse = npy.zeros(len(a))
            nrmse = npy.zeros(len(a))
            for i in xrange(0, len(a)):
               setting['exp_x'] = x_exp
               setting['exp_y'] = data[i*len(x_exp):(i+1)*len(x_exp), 2]
               y_exp = data[i*len(x_exp):(i+1)*len(x_exp), 2]
               PCE_lars = calPCE.collocation(n_deg, x_prob, x_exp, y_exp, 'LARS')
               setting['bounds'] = [(lower_bound, upper_bound), (lower_bound, upper_bound)]
               model = createKriging.createKriging(setting, PCE_lars)
               PCE_pred, y_PCE_pred = utilPCE.predictor(PCE_lars, x_pred)
               y_kriging, ci_kriging, var_kriging = createKriging.predictor(x_pred, model, setting, PCE_lars, PCE_pred)
               y_pred = y_kriging
               y_Real = y_real[i*len(x_pred):(i+1)*len(x_pred), 2]
               rmse[i], nrmse[i] = utilPCE.validation(y_Real, y_pred)
        return rmse, nrmse

def sobol_gen(a, x_prob, dim, n_exp, x_mix, *args):
    x_exp1 = args[0]
    x_exp2 = args[1]
    n_deg = args[2]
    x_exp = args[3]
    data = args[4]
    setting = args[5]
    upper_bound = args[6]
    lower_bound = args[7]
    y  = npy.zeros((n_exp, dim))
    y1 = npy.zeros((n_exp, 1))
    y2 = npy.zeros((n_exp, 1))
    setting['exp_x'] = x_exp
    setting['exp_y'] = data[:,2]
    y_exp = data[:,2]
    PCE_lars = calPCE.collocation(n_deg, x_prob, x_exp, y_exp, 'LARS')
    if len(a) == 0:
        setting['bounds'] = [(lower_bound, upper_bound), (lower_bound, upper_bound), (lower_bound, upper_bound)]
    else:
        setting['bounds'] = [(lower_bound, upper_bound), (lower_bound, upper_bound)]
    model = createKriging.createKriging(setting, PCE_lars)
    PCE_pred_1, y_PCE_pred = utilPCE.predictor(PCE_lars, x_exp1)
    y1, ci_kriging, var_kriging = createKriging.predictor(x_exp1, model, setting, PCE_lars, PCE_pred_1)
    PCE_pred_2, y_PCE_pred = utilPCE.predictor(PCE_lars, x_exp2)
    y2, ci_kriging, var_kriging = createKriging.predictor(x_exp2, model, setting, PCE_lars, PCE_pred_2)
    for i in xrange(0, dim):
        x_mix[:,:] = x_exp1
        x_mix[:,i] = x_exp2[:,i]
        PCE_pred, y_PCE_pred = utilPCE.predictor(PCE_lars, x_mix)
        y[:,i], ci_kriging, var_kriging = createKriging.predictor(x_mix, model, setting, PCE_lars, PCE_pred)

    return y1, y2, y