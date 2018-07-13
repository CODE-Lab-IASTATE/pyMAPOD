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

import numpy as npy
from scipy.stats import norm
import matplotlib.pyplot as plt

def pod_cal(data, beta0, beta1, tau, threshold):
    
    mu = (npy.log(threshold) - beta0) / beta1
    sigma = tau / beta1
    
    phi = npy.array([[1, 0], [mu, sigma], [0, -1]]) * (-1) / beta1
    cov_lr = cov_para3(data, tau)
    
    pcov = npy.matmul(npy.matmul(phi.transpose(), cov_lr), phi)
    
    return mu, sigma, pcov
    
    
def pod_para(mu, sigma, pcov):
    
    a_50 = npy.exp(norm.ppf(0.5, mu, sigma))
    a_90 = npy.exp(norm.ppf(0.9, mu, sigma))
    
    wp = norm.ppf(0.9, 0, 1)
    a_90_95 = pod_ci(pcov, a_90, wp)
    
    return a_50, a_90, a_90_95
    

def pod_view(mu, sigma, pcov):
    
    p = npy.linspace(0.001, 0.991, 100)
    a_pod = npy.exp(norm.ppf(p, mu, sigma))
    
    wp = norm.ppf(p, 0, 1)
    
    a_pod_95 = pod_ci(pcov, a_pod, wp)
    
    plt.figure()
    ax = plt.subplot(111)
    csfont = {'fontname':'Times New Roman',
              'fontsize':16}
    ax.set_xscale('log')
    # ax.set_xlim([min(a_pod), max(a_pod)])
    ax.plot(a_pod, p, 'k', linewidth=1)
    ax.plot(a_pod_95, p, 'k--', linewidth=0.5)
    ax.set_xlabel('Size, a (mm)', **csfont)
    ax.set_ylabel('Porbability of Detection, POD | a', **csfont)
    
    # ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # ax.set_xticklabels(['$10^{-1}$', 2, 3, 4, 5, 6, 7, 8, 9, '$10^0$'])
    for tick in ax.get_xticklabels():
        tick.set_fontname("Times New Roman")
        tick.set_fontsize(10)
    for tick in ax.get_yticklabels():
        tick.set_fontname("Times New Roman")
        tick.set_fontsize(10)
        
    text = "Key parameters: \n $ \\mu $ = %f, \n $ \\sigma $ = %f, \n covariance matrix: \n [%f %f \n ${ } { } { }$ %f %f]"%(mu, sigma, pcov[0,0], pcov[0,1], pcov[1,0], pcov[1,1])
    
    ax.text(min(a_pod), 0.5, text, style='italic',
            fontsize=10, fontname='Times New Roman',
            bbox={'facecolor':'white', 'alpha':0.0, 'pad':10})
   
    plt.show()

    
def cov_para3(data, tau):
    
    var0 = len(data) / tau**2
    var1 = sum(npy.log(data[:,1]) * npy.log(data[:,1])) / tau**2
    cov_para = sum(npy.log(data[:,1])) / tau**2
    var2 = 2*len(data) / tau**2
    
    FIM = npy.array([[var0, cov_para, 0], [cov_para, var1, 0], [0, 0, var2]])
    
    cov_lr = npy.linalg.inv(FIM)
    
    return cov_lr
    

def pod_ci(pcov, a, wp):
    
    sd = npy.sqrt(pcov[0,0] + wp*wp*pcov[1,1] + 2*wp*pcov[0,1])
    a_95 = npy.exp(npy.log(a) + 1.645*sd)
    
    return a_95