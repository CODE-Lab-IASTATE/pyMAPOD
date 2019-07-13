# Filename: sen_analysis
# Main function

# Created: November/09/2017
# Last modified: November/09/2017
# Author: Prof. Leifur Leifsson
#         PhD student: Xiaosong Du

import numpy as npy
import prob_distrs
import random
import func_data
import pyPCE
import pyKriging
import pyPCKriging

def exp_mix(a, x_prob, *args):

    x_exp1 = args[0]
    x_exp2 = args[1]
    dim = len(x_exp1[0,:])
    n_exp = len(x_exp1[:,0])
    x_mix  = npy.zeros((n_exp, dim))

    if len(args) == 4:
        y  = npy.zeros((n_exp, dim))
        y1 = npy.zeros((n_exp, 1))
        y2 = npy.zeros((n_exp, 1))
        y1_data = npy.zeros((n_exp, 3))
        y2 = npy.zeros((n_exp, 3))
        funcForm = args[2]
        full_model = args[3]
        y1_data = func_data.run_func(funcForm, full_model, a, x_exp1)
        y1 = y1_data[:,2]
        y2_data = func_data.run_func(funcForm, full_model, a, x_exp2)
        y2 = y2_data[:,2]
    
        for i in xrange(0, dim):
            x_mix[:,:] = x_exp1
            x_mix[:,i] = x_exp2[:,i]
            y_data = func_data.run_func(funcForm, full_model, a, x_mix)
            y[:,i] = y_data[:,2]
    elif len(args) == 6:
        y1, y2, y = pyPCE.sobol_gen(a, x_prob, dim, n_exp, x_mix, *args)
    elif len(args) == 7:
        y1, y2, y = pyKriging.sobol_gen(a, x_prob, dim, n_exp, x_mix, *args)
    elif len(args) == 8:
        y1, y2, y = pyPCKriging.sobol_gen(a, x_prob, dim, n_exp, x_mix, *args)
    return y1, y2, y
    
def Sobol_1st(y1, y2, y_exp):
    
    V = npy.zeros((len(y_exp[0,:]), 1))
    for i in xrange(0, len(y_exp[0,:])):
        V[i] = 1.0/len(y_exp[:,0])*npy.sum(y2*(y_exp[:,i]-y1))
    return V
    
def Sobol_total(y1, y2, y_exp):
    
    V = npy.zeros((len(y_exp[0,:]), 1))
    for i in xrange(0, len(y_exp[0,:])):
        V[i] = 0.5/len(y_exp[:,0])*npy.sum((y_exp[:,i]-y1)**2)
    return V

def Sobol(a, x_prob, conf = False, resample = 0, conf_inter = 0, *args):
    
    [y1, y2, y_exp] = exp_mix(a, x_prob, *args)

    # for 1st-order Sobol
    V_1st = Sobol_1st(y1, y2, y_exp)
    
    # for total-order Sobol
    V_total = Sobol_total(y1, y2, y_exp)
    
    # Variance of y
    y = npy.concatenate((y1, y2), axis = 0)
    for i in xrange(0, len(x_prob)):
        y = npy.concatenate((y, y_exp[:,i]), axis = 0)
    
    if conf == True:
        V_1st_resample = npy.zeros((len(x_prob), 1, resample))
        V_total_resample = npy.zeros((len(x_prob), 1, resample))
        y1_resample = npy.zeros(len(y1))
        y2_resample = npy.zeros(len(y1))
        y_resample  = npy.zeros((len(y1), len(x_prob)))
        V_1st_conf = npy.zeros((len(x_prob), 2))
        V_total_conf = npy.zeros((len(x_prob), 2))
        for i in xrange(0, resample):
            # x_sample_resample = {'LHS' : len(y1)}
            # x_prob_resample = {"Uniform" : (0, len(y1))}
            # x_exp = prob_distrs.gen_exp(x_sample_resample, x_prob_resample)
            for j in xrange(0, len(y1)):
                selection = random.choice(npy.arange(0, len(y1)))
                x_exp = int(selection)
                y1_resample[j] = y1[x_exp]
                y2_resample[j] = y2[x_exp]
                y_resample[j,:]  = y_exp[int(x_exp), :]

            V_1st_resample[:, :, i] = Sobol_1st(y1_resample, y2_resample, y_resample)
            V_total_resample[:, :, i] = Sobol_total(y1_resample, y2_resample, y_resample)
        
        for i in xrange(0, len(x_prob)):
            V_1st_sort = npy.sort(V_1st_resample[i,0,:])
            V_total_sort = npy.sort(V_total_resample[i,0,:])           
            V_1st_conf[i,0] = V_1st_sort[int(resample*conf_inter)]
            V_1st_conf[i,1] = V_1st_sort[int(resample*(1-conf_inter))]
            V_total_conf[i,0] = V_total_sort[int(resample*conf_inter)]
            V_total_conf[i,1] = V_total_sort[int(resample*(1-conf_inter))]

        V_1st = npy.concatenate((V_1st, V_1st_conf), axis = 1)
        V_total = npy.concatenate((V_total, V_total_conf), axis = 1)

    return V_1st/npy.var(y), V_total/npy.var(y)
