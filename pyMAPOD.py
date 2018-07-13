######################################################################
#                                                                    #
#                                                                    #
#             Model-Assisted Probability of Detection                #
#                                                                    #
#                       Open-Source Framework                        #
#                                                                    #
#           Developed by:  Computation Design (CODE) Lab             #
#                                                                    #
#                                                                    #
######################################################################

import func_data
import ahat_vs_a
import pod_gen
import collections
import prob_distrs
import numpy as npy
import sparse_md
import test_func

# use existing data or model function
input_type = 'func'   # 'dat' for existing data
                      # 'func' for linking simulation model

if input_type == 'dat':                   
    # the name of file containing database
    fileName = 'test_MAPOD.xlsx'
    # the name of sheet of interest
    sheetName = 'F_theta_x'
    # extract and view data
    data = func_data.read_view_data(fileName, sheetName, view_data = True)

elif input_type == 'func':
    # the format of function for simulation model
    funcForm = 'python' # options: python, matlab
    # the name of model function
    funcName = test_func.simpleFunc

    # funcForm = 'matlab' # options: python, matlab
    # funcName = 'test_func_mat'

    # define a series of "a" (i.e. defect size) values
    # or "a" can be used as first uncertainty parameter among random inputs
    # a = npy.array([0.1, 0.2, 0.3, 0.4, 0.5])
    a = npy.array([])
    # define random inputs with corresponding probability distributions                               
    x_prob = collections.OrderedDict([('Uniform1', (0.1, 0.5)),
                                      ('Uniform2', (3, 4)),
                                      ('Gaussian1', (5, 0.5))])
    x_sample = {'LHS' : 100}     # 'LHS': Latin Hypercube Sampling
                                 # 'MCS': Monte Carlo Sampling 
                                 # value: number of sample points
    # generate sample points
    x_exp = prob_distrs.gen_exp(x_sample, x_prob)
    # run simulation models on sample points
    data = func_data.run_func(funcForm, funcName, a, x_exp)

    # decide whether generate PCE metamodel for real physics model
    PCE_gen = True
    if PCE_gen == True:
        # the order of PCE
        n_deg = 7
        # the number of points for prediction
        x_meta = {'LHS' : 1000}
        x_pred = prob_distrs.gen_exp(x_meta, x_prob)
        data, statistics = sparse_md.meta_gen(PCE_gen, n_deg, x_pred, x_exp, data, x_prob, a)
    else:
        pass

# "ahat vs. a" regression
beta0, beta1, tau = ahat_vs_a.regression(data)
# view regression
ahat_vs_a.view_reg(data, beta0, beta1, tau)

# user-defined detection threshold, units vary due to the type of model response
threshold = 6.5
# pod generation
mu, sigma, pcov = pod_gen.pod_cal(data, beta0, beta1, tau, threshold)
# key parameters
a_50, a_90, a_90_95 = pod_gen.pod_para(mu, sigma, pcov)
# view pod curves
pod_gen.pod_view(mu, sigma, pcov)