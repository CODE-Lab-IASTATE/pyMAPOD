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
import matplotlib.pyplot as plt

def regression(data):
    
    x = npy.log(data[:, 1])
    y = npy.log(data[:, 2])
    
    beta1 = sum((x-npy.mean(x))*(y-npy.mean(y))) / sum((x-npy.mean(x))*(x-npy.mean(x)))
    beta0 = npy.mean(y) - beta1*npy.mean(x)
    tau = npy.sqrt( sum((y - (beta0 + beta1*x)) * (y - (beta0 + beta1*x))) / len(data) )
    
    return beta0, beta1, tau
    
    
def view_reg(data, beta0, beta1, tau):
    
    pcov = cov_para2(data, tau)
    
    x_min = npy.log(min(data[:, 1]))
    x_max = npy.log(max(data[:, 1]))
    x_lin = npy.linspace(x_min, x_max, 100)
    y_lin = beta0 + beta1 * x_lin
    
    var_y = pcov[0,0] + 2*x_lin*pcov[0,1] + x_lin*x_lin*pcov[1,1]
    var_total = var_y + tau**2
    
    y_lin_lb = beta0 + beta1 * x_lin - 1.645*npy.sqrt(var_y)
    y_lin_ub = beta0 + beta1 * x_lin + 1.645*npy.sqrt(var_y)
    y_lin_lb_total = beta0 + beta1 * x_lin - 1.645*npy.sqrt(var_total)
    y_lin_ub_total = beta0 + beta1 * x_lin + 1.645*npy.sqrt(var_total)
    
    
    plt.figure()
    ax = plt.subplot(111)
    ax.plot(data[:,1], data[:,2], 'ks', markersize=0.5)
    ax.plot(npy.exp(x_lin), npy.exp(y_lin), 'k', linewidth=1)
    ax.plot(npy.exp(x_lin), npy.exp(y_lin_lb), 'b--', linewidth=0.5)
    ax.plot(npy.exp(x_lin), npy.exp(y_lin_ub), 'b--', linewidth=0.5)
    ax.plot(npy.exp(x_lin), npy.exp(y_lin_lb_total), 'b--', linewidth=0.5)
    ax.plot(npy.exp(x_lin), npy.exp(y_lin_ub_total), 'b--', linewidth=0.5)
    csfont = {'fontname':'Times New Roman',
            'fontsize':16}
    ax.set_xlabel("Size, a (mm)", **csfont)
    ax.set_ylabel("Response, $\hat{a}$ (mV)", **csfont)
    ax.set_yscale('log')
    ax.set_xscale('log')
    
#   ax.set_xlim([0.09, 1.02])
#   ax.set_ylim([0.009, 101])
    
    for tick in ax.get_xticklabels():
        tick.set_fontname("Times New Roman")
        tick.set_fontsize(10)
    for tick in ax.get_yticklabels():
        tick.set_fontname("Times New Roman")
        tick.set_fontsize(10)
    
    text = "Key parameters: \n $ \\beta_0 $ = %f, \n $ \\beta_1 $ = %f, \n $ \\tau $ = %f"%(beta0, beta1, tau)
    
    ax.text(min(data[:,1]), npy.mean(data[:,2]), text, style='italic',
            fontsize=10, fontname='Times New Roman',
            bbox={'facecolor':'white', 'alpha':0.0, 'pad':10})

    plt.show()


def cov_para2(data, tau):
    
    var0 = len(data)/tau**2
    var1 = sum(npy.log(data[:,1]) * npy.log(data[:,1]))/tau**2
    cov_para = sum(npy.log(data[:,1]))/tau**2
    
    FIM = npy.array([[var0, cov_para], [cov_para, var1]])
    
    pcov = npy.linalg.inv(FIM)
    
    return pcov