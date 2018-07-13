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

import pandas
import matplotlib.pyplot as plt
import sys
import numpy as npy
import os

def read_view_data(fileName, sheetName, view_data):
    
    # import database based on the name of Excel file and corresponded sheet
    data_base = pandas.read_excel(fileName, sheetname = sheetName)
    
    # extract data from imported database
    data = data_base.as_matrix()
    
    # view data in log scale
    if view_data:
        plt.figure()
        ax = plt.subplot(111)
        ax.plot(data[:,1], data[:,2], 'ks', markersize = 0.5)
        csfont = {'fontname':'Times New Roman',
                'fontsize':16}
        ax.set_xlabel("Size, a (mm)", **csfont)
        ax.set_ylabel("Response, $\hat{a}$ (mV)", **csfont)
        ax.set_yscale('log')
        ax.set_xscale('log')
    
        for tick in ax.get_xticklabels():
            tick.set_fontname("Times New Roman")
            tick.set_fontsize(10)
        for tick in ax.get_yticklabels():
            tick.set_fontname("Times New Roman")
            tick.set_fontsize(10)
    
        plt.show()
    
    
    return data
    
def run_func(funcForm, funcName, a, x_exp):
    
    if funcForm == 'python':
        if len(a) == 0:
            data = npy.zeros([len(x_exp), 3])
            for i in xrange(0, len(x_exp)):
                rsp = funcName(x_exp[i,0], x_exp[i,1:])
                data[i,:] = npy.array([i+1, x_exp[i,0], rsp])
        else:
            data = npy.zeros([len(a)*len(x_exp), 3])
            for i in xrange(0, len(a)):
                for j in xrange(0, len(x_exp)):
                    rsp = funcName(a[i], x_exp[j,:])
                    data[i*len(x_exp)+j,:] = npy.array([i*len(x_exp)+j+1, a[i], rsp])
        
    elif funcForm == 'matlab':
        import mlab
        from mlab.releases import latest_release as matlab
        if len(a) == 0:
            data = npy.zeros([len(x_exp), 3])
            for i in xrange(0, len(x_exp)):
                rsp = matlab.cfg_modelFunc_pyMAPOD(x_exp[i,0], x_exp[i,1:], funcName)
                data[i,:] = npy.array([i+1, x_exp[i,0], rsp])
        else:
            data = npy.zeros([len(a)*len(x_exp), 3])
            for i in xrange(0, len(a)):
                for j in xrange(0, len(x_exp)):
                    rsp = matlab.cfg_modelFunc_pyMAPOD(a[i], x_exp[j,:], funcName)
                    data[i*len(x_exp)+j,:] = npy.array([i*len(x_exp)+j+1, a[i], rsp])  
        
       
    else:
        print 'cannot recognize such format of simulation model'
        print 'now exit!'
        sys.exit()
        
    return data