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
import test_func
import pyPCE
import pyKriging
import pyPCKriging
import sens_analysis


# use existing data or model function
input_type = 'dat'   # 'dat' for existing data
                      # 'func' for linking simulation model


#The following 3 options are to be when the other 2 are turned off (False)

#Use this option when:
#val_metamodel = False and calc_sobol = False
#generate the metamodel for prediction
gen_metamodel = False

#Use this option when:
#gen_metamodel = False and calc_sobol = False
#Validate the metamodel 
#Only use this if you generate the metamodel
val_metamodel = False

#Use this option when:
#gen_metamodel = False and val_metamodel = False
#Calculate Sobol indices
calc_sobol = False
#Calculate Sobol indices using a metamodel
#This option can only be used if calc_sobol = True
calc_sobol_meta = False


if input_type == 'dat':                   
    # the name of file containing database
    fileName = 'test_MAPOD.xlsx'
    # the name of sheet of interest
    sheetName = 'F_theta_x'
    # extract and view data
    data = func_data.read_view_data(fileName, sheetName, view_data = True)

elif input_type == 'func':
    # the format of function for simulation model

    #Python
    funcForm = 'python' # options: python, matlab
    # the name of model function
    funcName = test_func.simpleFunc

    #Matlab
    # funcForm = 'matlab' # options: python, matlab
    # funcName = 'test_func_mat'

    # define a series of "a" (i.e. defect size) values
    # or "a" can be used as first uncertainty parameter among random inputs
    # a: discrete or continuous
    a_type = 'discrete'

    if a_type == 'discrete':
        a = npy.array([0.1, 0.2, 0.3, 0.4, 0.5])
        x_prob = collections.OrderedDict([('Uniform1', (3, 4)),
                                          ('Gaussian1', (5, 0.5))])
    elif a_type == 'continuous':
        a = npy.array([])
        # define random inputs with corresponding probability distributions                               
        x_prob = collections.OrderedDict([('Uniform1', (0.1, 0.5)),
                                          ('Uniform2', (3, 4)),
                                          ('Gaussian1', (5, 0.5))])


    x_sample = {'LHS' : 120}    # 'LHS': Latin Hypercube Sampling
                                 # 'MCS': Monte Carlo Sampling 
                                 # value: number of sample points
    # generate sample points
    x_exp = prob_distrs.gen_exp(x_sample, x_prob)
 
    # run simulation models on sample points
    data = func_data.run_func(funcForm, funcName, a, x_exp)

    if calc_sobol == True:
        x_sobol = {'LHS' : 100}
        x_exp1 = prob_distrs.gen_exp(x_sobol, x_prob)
        x_exp2 = prob_distrs.gen_exp(x_sobol, x_prob)
        if calc_sobol_meta == False:
            if len(a) == 0:
                [Sobol_1st, Sobol_total] = sens_analysis.Sobol(a, x_prob, True, 1000, 0.05, x_exp1, x_exp2, funcForm, funcName)
            else:
                sobol_1st_1 = npy.zeros((len(a),3))
                sobol_1st_2 = npy.zeros((len(a),3))
                sobol_total_1 = npy.zeros((len(a),3))
                sobol_total_2 = npy.zeros((len(a),3))
                for i in xrange(0, len(a)):
                    [Sobol_1st, Sobol_total] = sens_analysis.Sobol([a[i]], x_prob, True, 1000, 0.05, x_exp1, x_exp2, funcForm, funcName)
                    sobol_1st_1[i,:] = Sobol_1st[0,:]
                    sobol_1st_2[i,:] = Sobol_1st[1,:]
                    sobol_total_1[i,:] = Sobol_total[0,:]
                    sobol_total_2[i,:] = Sobol_total[1,:]

    # decide which metamodel to use
    PCE_gen = False
    Kriging_gen = False
    PCKriging_gen = True

    if gen_metamodel == True:
        # the number of points for prediction
        x_meta = {'LHS' : 1000}
        x_pred = prob_distrs.gen_exp(x_meta, x_prob)
    elif val_metamodel == True:
        # the number of points for validation
        x_vali = {'MCS' : 1000}
        x_val = prob_distrs.gen_exp(x_vali, x_prob)
        y_real = func_data.run_func(funcForm, funcName, a, x_val)

    if PCE_gen == True:
        # the order of PCE
        n_deg = 7
        # the type of scheme to solve for PCE coefficients
        # OLS: ordinary least squares; LARS: least angle regression
        meta_type = 'LARS'
        if gen_metamodel == True:
            data = pyPCE.meta_gen(n_deg, x_pred, x_exp, data, x_prob, a, meta_type)
        elif val_metamodel == True:
            rmse, nrmse = pyPCE.meta_val(n_deg, x_val, y_real, x_exp, data, x_prob, a, meta_type)
        elif calc_sobol_meta == True:
            if len(a) == 0:
                [Sobol_1st, Sobol_total] = sens_analysis.Sobol(a, x_prob, True, 1000, 0.05, x_exp1, x_exp2, n_deg, x_exp, data, meta_type)
            else:
                sobol_1st_1 = npy.zeros((len(a),3))
                sobol_1st_2 = npy.zeros((len(a),3))
                sobol_total_1 = npy.zeros((len(a),3))
                sobol_total_2 = npy.zeros((len(a),3))
                for i in xrange(0, len(a)):
                    y_exp = data[i*len(x_exp):(i+1)*len(x_exp), :]

                    [Sobol_1st, Sobol_total] = sens_analysis.Sobol([a[i]], x_prob, True, 1000, 0.05, x_exp1, x_exp2, n_deg, x_exp, y_exp, meta_type)
                    sobol_1st_1[i,:] = Sobol_1st[0,:]
                    sobol_1st_2[i,:] = Sobol_1st[1,:]
                    sobol_total_1[i,:] = Sobol_total[0,:]
                    sobol_total_2[i,:] = Sobol_total[1,:]

    elif Kriging_gen == True:
        # settings
        setting = {}
        # trend function type
        setting['trendType'] = 'linear'
        # setting['trendDegree'] = 3 (only used for "polynomial" type)
        # correlation function type
        setting['corrType'] = 'ellipsoidal_aniso'
        # correlation function family
        setting['corrFam'] = 'gaussian'
        # nuggest
        setting['nugget'] = 1e-10
        # estimation method
        setting['estimator'] = 'maximum_likelihood'#'cross_validation'
        # optimization method
        lower_bound = 1e-10
        upper_bound = 10
        setting['optimizer'] = 'SLSQP'
        if gen_metamodel == True:
            data = pyKriging.meta_gen(x_pred, x_exp, data, x_prob, a, setting, lower_bound, upper_bound)
        elif val_metamodel == True:
            rmse, nrmse = pyKriging.meta_val(x_val, y_real, x_exp, data, x_prob, a, setting, lower_bound, upper_bound)
        elif calc_sobol_meta == True:
            if len(a) == 0:
                [Sobol_1st, Sobol_total] = sens_analysis.Sobol(a, x_prob, True, 1000, 0.05, x_exp1, x_exp2, x_exp, data, setting, lower_bound, upper_bound)
            else:
                sobol_1st_1 = npy.zeros((len(a),3))
                sobol_1st_2 = npy.zeros((len(a),3))
                sobol_total_1 = npy.zeros((len(a),3))
                sobol_total_2 = npy.zeros((len(a),3))
                for i in xrange(0, len(a)):
                    y_exp = data[i*len(x_exp):(i+1)*len(x_exp), :]
                    [Sobol_1st, Sobol_total] = sens_analysis.Sobol([a[i]], x_prob, True, 1000, 0.05, x_exp1, x_exp2, x_exp, y_exp, setting, lower_bound, upper_bound)
                    sobol_1st_1[i,:] = Sobol_1st[0,:]
                    sobol_1st_2[i,:] = Sobol_1st[1,:]
                    sobol_total_1[i,:] = Sobol_total[0,:]
                    sobol_total_2[i,:] = Sobol_total[1,:]
    elif PCKriging_gen == True:
        # the order of PCE
        n_deg = 8
        # settings
        setting = {}
        # trend function type
        setting['trendType'] = 'pce'
        # correlation function type
        setting['corrType'] = 'ellipsoidal_aniso'
        # correlation function family
        setting['corrFam'] = 'matern_5_2'
        # nuggest
        setting['nugget'] = 1e-10
        # estimation method
        lower_bound = 1e-10
        upper_bound = 10
        setting['estimator'] = 'maximum_likelihood'
        # optimization method
        setting['optimizer'] = 'SLSQP' 
        if gen_metamodel == True:
            data = pyPCKriging.meta_gen(n_deg, x_pred, x_exp, data, x_prob, a, setting, lower_bound, upper_bound)
        elif val_metamodel == True:
            rmse, nrmse = pyPCKriging.meta_val(n_deg, x_val, y_real, x_exp, data, x_prob, a, setting, lower_bound, upper_bound)
        elif calc_sobol_meta == True:
            if len(a) == 0:
                [Sobol_1st, Sobol_total] = sens_analysis.Sobol(a, x_prob, True, 1000, 0.05, x_exp1, x_exp2, n_deg, x_exp, data, setting, lower_bound, upper_bound)
            else:
                sobol_1st_1 = npy.zeros((len(a),3))
                sobol_1st_2 = npy.zeros((len(a),3))
                sobol_total_1 = npy.zeros((len(a),3))
                sobol_total_2 = npy.zeros((len(a),3))
                for i in xrange(0, len(a)):
                    y_exp = data[i*len(x_exp):(i+1)*len(x_exp), :]
                    [Sobol_1st, Sobol_total] = sens_analysis.Sobol([a[i]], x_prob, True, 1000, 0.05, x_exp1, x_exp2, n_deg, x_exp, y_exp, setting, lower_bound, upper_bound)
                    sobol_1st_1[i,:] = Sobol_1st[0,:]
                    sobol_1st_2[i,:] = Sobol_1st[1,:]
                    sobol_total_1[i,:] = Sobol_total[0,:]
                    sobol_total_2[i,:] = Sobol_total[1,:]
    else:
        pass

if val_metamodel == True:
    if len(a) == 0:
        print("The RMSE is: " + str(rmse))
        print("\n")
        print("The NRMSE is: " + str(nrmse))
    else:
        for i in xrange(0, len(a)):
            print("Defect size "+ str(a[i]))
            print("The RMSE: " + str(rmse[i]))
            print("The NRMSE: "+ str(nrmse[i]))
            print("\n")


if val_metamodel == False and calc_sobol == False:
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

if calc_sobol == True:
    if len(a) == 0:
        print("1st order Sobol index")
        print("1st uncertainty parameter: " + str(Sobol_1st[0,0]))
        print("2nd uncertainty parameter: " + str(Sobol_1st[1,0]))
        print("\n")
        print("Total order Sobol index")
        print("1st uncertainty parameter: " + str(Sobol_total[0,0]))
        print("2nd uncertainty parameter: " + str(Sobol_total[1,0]))

    else:
        for i in xrange(0, len(a)):
            print("Defect size "+ str(a[i]))
            print("1st order Sobol index")
            print("1st uncertainty parameter: " + str(sobol_1st_1[i,0]))
            print("2nd uncertainty parameter: " + str(sobol_1st_2[i,0]))
            print("\n")
            print("Total order Sobol index")
            print("1st uncertainty parameter: " + str(sobol_total_1[i,0]))
            print("2nd uncertainty parameter: " + str(sobol_total_2[i,0]))
            print("\n")


