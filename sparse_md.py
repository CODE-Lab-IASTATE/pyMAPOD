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
import sys
from sklearn import linear_model
import math

def multichoose(n,k):
    '''input arguments:
       n: dimension of random inputs
       k: kth order of PCE, note: not all orders provided
    '''
    if k < 0 or n < 0: return "Error"
    if not k: return [[0]*n]
    if not n: return []
    if n == 1: return [[k]]
    return [[val[0]+1]+val[1:] for val in multichoose(n,k-1)] + \
            [[0]+val for val in multichoose(n-1,k)]
            
def iter_basis(basis, x_prob, n_deg, index):
    
    basis_total = npy.array(([]))
    
    if len(basis[0,0,:]) == 1:
        return basis[:,int(index[0]),-1]
        
    for i in xrange(0, n_deg+1):
        basis_total = npy.append(basis_total, basis[i,int(index[0]),0]*iter_basis(basis[:,:,1:], x_prob, n_deg, index[1:]))
              
    return basis_total

def iter_weights(weight, x_prob, n_deg):

    weight_total = npy.array(([]))

    if len(weight[0,:]) == 1:
        return weight[:,-1]
        
    for i in xrange(0, n_deg+1):
        weight_total = npy.append(weight_total, weight[i,0]*iter_weights(weight[:,1:], x_prob, n_deg))
        
    return weight_total
    
def gen_legendre_md(basis, x, n_deg):
    basis[:,0] = 1
    basis[:,1] = x
    for i in xrange(1, n_deg):
        basis[:,i+1] = ((2*i+1)*x*basis[:,i] - i*basis[:,i-1]) / (i+1.0)
    
    return basis

def gen_hermite_md(basis, x, n_deg):
    basis[:,0] = 1
    basis[:,1] = x
    
    for i in xrange(1, n_deg):
        basis[:,i+1] = x*basis[:,i] - i*basis[:,i-1]
        
    return basis

def coef_legendre_md(x, n_deg):
    
    basis = npy.zeros((len(x), n_deg+1))
    
    basis[:,0] = 1
    basis[:,1] = npy.transpose(x)
    
    for i in xrange(1, n_deg):
        basis[:,i+1] = ((2*i+1)*npy.transpose(x)*basis[:,i] - i*basis[:,i-1]) / (i+1.0)
    
    return basis
    
def coef_hermite_md(x, n_deg):
    
    basis = npy.zeros((len(x), n_deg+1))
    
    basis[:,0] = 1
    basis[:,1] = npy.transpose(x)
    
    for i in xrange(1, n_deg):
        basis[:,i+1] = npy.transpose(x)*basis[:,i] - i*basis[:,i-1]
        
    return basis

def gen_basis_md(n_deg, key, value, x_experiment, full_model=npy.array([])):
    
    '''inputs:
       n_deg: required degree of PCE, scalar
       key: distribution type of random variable, string
       value: parameter of the distribution, tuple
       full_model: real model for evaluation use
       outputs:
       basis: basis of PCE, npy.ndarray
    '''
    
#     [x, weight] = spy.special.orthogonal.p_roots(n_deg+1)
#     weight = weight / 2 # thus, sum(weight) = 1

    if key.startswith('Uniform'):
        [x, weight] = npy.polynomial.legendre.leggauss(n_deg+1)
    elif key.startswith('Gaussian'):
        [x, weight] = npy.polynomial.hermite.hermgauss(n_deg+1)
        
        x = npy.sqrt(2) * x
        
    else:
        print 'Other statistical distributions are under construction!'
        print 'Now exiting!'
        sys.exit()
    
    weight = weight / sum(weight) # thus, sum(weight) = 1
    
    basis = npy.zeros((n_deg+1, len(x)))
    
    if key.startswith('Uniform'):
        
        print 'Current distribution is uniform, legendre basis will be used'        
        
        if n_deg == 0:
            print 'Required order of PCE is 0, which is no applicable'
        elif n_deg == 1:
            basis[:,0] = 1
            basis[:,1] = x
        else:
            basis = gen_legendre_md(basis,x,n_deg)
            
        x = 0.5*(value[1]-value[0])*x + 0.5*(value[1]+value[0]) # scale it from [-1, 1]
        
        if len(full_model) != 0:
            y_real = full_model(x)
        else:
            y_real = npy.array([])
        return basis, weight, y_real
        
    elif key.startswith('Gaussian'):
        
        print 'Current distribution is Gaussian, Hermite basis will be used'
        
        if n_deg == 0:
            print 'Required order of PCE is 0, which is no applicable'
        elif n_deg == 1:
            basis[:,0] = 1
            basis[:,1] = x
        else:
            basis = gen_hermite_md(basis,x,n_deg)
        
        x = value[0] + value[1] * x
        
        if len(full_model) != 0:
            y_real = full_model(x)
        else:
            y_real = npy.array([])        
                             
        return basis, weight, y_real
            
    else:
        
        print 'Other types of distributions are under construction!'
        print 'EXiting now'
        sys.exit()

def sparse_md(n_deg, x_prob, x_experiment, y_experiment, meta_model):
    
    '''inputs:
       n_deg: required degree of PCE, scalar
       x_prob: random-variable information, dictionary
       x_experiment: random inputs based on statistical distributions
       y_experiment: response from real model, corresponded with x_experiment
       outputs:
       alpha: coefficients of PCE, npy.ndarray
       phi: basis of PCE npy.ndarray
    '''
    
    # generate basis function first
    i = 0
    basis = npy.zeros((n_deg+1, n_deg+1, len(x_prob)))
    y_real= npy.zeros((n_deg+1, len(x_prob)))
    
    weight= npy.zeros((n_deg+1, len(x_prob)))
    basis_coef = npy.zeros((len(x_experiment), n_deg+1, len(x_prob)))
    
    alpha = npy.zeros((math.factorial(n_deg+len(x_prob))/math.factorial(n_deg)/math.factorial(len(x_prob)),1))
    Nk = npy.zeros((math.factorial(n_deg+len(x_prob))/math.factorial(n_deg)/math.factorial(len(x_prob)),1))
    
    basis_total = npy.ones(((n_deg+1)**len(x_prob), math.factorial(n_deg+len(x_prob))/math.factorial(n_deg)/math.factorial(len(x_prob)), 1))
    basis_coef_total = npy.ones((len(x_experiment), math.factorial(n_deg+len(x_prob))/math.factorial(n_deg)/math.factorial(len(x_prob)), 1))
    
       
    for key, value in x_prob.iteritems():
        
        [basis[:,:,i], weight[:,i], y_real] = gen_basis_md(n_deg, key, value, x_experiment)
        
        # Nk[:,i] = npy.matmul(weight, basis[:,:,i]*basis[:,:,i])
        
        if key.startswith('Uniform'):
            lb = value[0]
            ub = value[1]
            x = (2.0*x_experiment[:,i] - ub - lb) / (ub - lb) 
            
            basis_coef[:,:,i] = coef_legendre_md(x, n_deg)

        elif key.startswith('Gaussian'):
            location = value[0]
            scale    = value[1]
            x = (x_experiment[:,i] - location) / scale
            
            basis_coef[:,:,i] = coef_hermite_md(x, n_deg)
            
        else:
            print 'sparse_md.sparse_md here'
            print 'Other statistical distributions are still under constructions'
            print 'Now exiting'
            sys.exit()
            
        i = i + 1
    
    index = npy.array([])
    for iters in xrange(0, n_deg+1):
        index = npy.append(index, multichoose(len(x_prob), iters))

    for iters in xrange(0, len(index)/len(x_prob)):
        index_cal = index[iters*len(x_prob):(iters+1)*len(x_prob)]
        basis_total[:,iters,0] = iter_basis(basis, x_prob, n_deg, index_cal)
    
    for i in xrange(0, len(x_experiment)):
        for iters in xrange(0, math.factorial(n_deg+len(x_prob))/math.factorial(n_deg)/math.factorial(len(x_prob))):
            for j in xrange(0, len(x_prob)):
                basis_coef_total[i,iters,0] = basis_coef_total[i,iters,0]*basis_coef[i,int(index[iters*len(x_prob)+j]),j]
                
    weight_total = iter_weights(weight, x_prob, n_deg)
            
    Nk[:,0] = npy.matmul(weight_total, basis_total[:,:,0]*basis_total[:,:,0])
    
    if meta_model == 'OLS':
        reg = linear_model.LinearRegression()
        reg.__init__(fit_intercept=False, normalize=True, copy_X=True, n_jobs=1)
        
        reg.fit(basis_coef_total[:,:,0], y_experiment)
            
        alpha[:,0] = reg.coef_
            
    elif meta_model == 'LARS':

        # perform sparse PCE
        # options: lasso(L1 norm), lar(L2 norm)
        # alphas, _, coefs = linear_model.lars_path(basis_coef, y_experiment[:,0], method='lasso', verbose=False, positive=False)
        # alpha[:,i] = coefs[:,-1]
            
        reg = linear_model.LassoLarsCV()
        reg.__init__(fit_intercept=True, verbose=False, normalize=True, copy_X=True, positive=False)
        
        reg.fit(basis_coef_total[:,:,0], y_experiment)
        alpha[:,0] = reg.coef_
        alpha[0,0] = reg.intercept_

    else:
        print 'no such type of surrogate method found'
        print 'now exiting!'
        sys.exit()
        
    return alpha, Nk, basis_coef_total
    
def prediction_legendre_md(x, n_deg):
    
    basis = npy.zeros((len(x), n_deg+1))
    
    basis[:,0] = 1
    basis[:,1] = x
    
    for i in xrange(1, n_deg):
        basis[:,i+1] = ((2*i+1)*x*basis[:,i] - i*basis[:,i-1]) / (i+1.0)
    
    return basis
    
def prediction_hermite_md(x, n_deg):
    
    basis = npy.zeros((len(x), n_deg+1))
    
    basis[:,0] = 1
    basis[:,1] = x
      
    for i in xrange(1, n_deg):
        basis[:,i+1] = x*basis[:,i] - i*basis[:,i-1]
    
    return basis
    
    
def prediction_md(x, n_deg, alpha, Nk, x_prob):
    
    i = 0
    basis = npy.zeros((len(x), n_deg+1, len(x_prob)))
    
    basis_total = npy.ones((len(x), math.factorial(n_deg+len(x_prob))/math.factorial(n_deg)/math.factorial(len(x_prob))))
    
    for key, value in x_prob.iteritems():
        
        if key.startswith('Uniform'):

            lb = value[0]
            ub = value[1]
            x_coef = (2.0*x[:,i] - ub - lb) / (ub - lb)
            
            basis[:,:,i] = prediction_legendre_md(x_coef, n_deg)
            
        elif key.startswith('Gaussian'):
            x_coef = (x[:,i] - value[0]) / value[1]
            basis[:,:,i] = prediction_hermite_md(x_coef, n_deg)
            
        else:
            print 'Other statistical distributions are under construction'
            print 'now exiting'
            sys.exit()
        
        i = i + 1
    
    index = npy.array([])
    for iters in xrange(0, n_deg+1):
        index = npy.append(index, multichoose(len(x_prob), iters))
    
    for i in xrange(0, len(x)):
        for iters in xrange(0, math.factorial(n_deg+len(x_prob))/math.factorial(n_deg)/math.factorial(len(x_prob))):
            for j in xrange(0, len(x_prob)):
                basis_total[i,iters] = basis_total[i,iters]*basis[i,int(index[iters*len(x_prob)+j]),j]
    
    y_predict = (alpha * npy.transpose(basis_total)).sum(0)

    return y_predict
    
def stats_md(alpha, Nk):
    
    mean = alpha[0] / npy.sqrt(Nk[0])
    std  = npy.sqrt(sum((alpha[1:]*npy.sqrt(Nk[1:]))**2)) 
    
    stats = npy.array([mean, std])
    
    return stats
    
def meta_gen(PCE_gen, n_deg, x_pred, x_exp, data, x_prob, a):
    
    if len(a) == 0:
        x_experiment = x_exp
        y_experiment = data[:, 2]
        [alpha, Nk, basis] = sparse_md(n_deg, x_prob, x_experiment, y_experiment, 'LARS')
        statistics = stats_md(alpha, Nk)
        y_pred = (prediction_md(x_pred, n_deg, alpha, Nk, x_prob)).reshape(len(x_pred),1)
        y_pred[y_pred<0.0001] = min(y_pred[y_pred>0.0001])
        index = npy.arange(1, len(x_pred)+1)
        index.shape = (len(x_pred),1)
        x_a = x_pred[:,0].reshape(len(x_pred),1)
        data_meta = npy.concatenate((index, x_a, y_pred), axis = 1)
    else:
        data_meta = npy.zeros((len(a)*len(x_pred), 3))
        statistics= npy.zeros((len(a), 2))
        for i in xrange(0, len(a)):
            x_experiment = x_exp
            y_experiment = data[i*len(x_exp):(i+1)*len(x_exp), 2]
            [alpha, Nk, basis] = sparse_md(n_deg, x_prob, x_experiment, y_experiment, 'LARS')
            statistics[i, :] = (stats_md(alpha, Nk)).reshape(2)
            y_pred = (prediction_md(x_pred, n_deg, alpha, Nk, x_prob)).reshape(len(x_pred),1)
            y_pred[y_pred<0.0001] = min(y_pred[y_pred>0.0001])
            index = npy.arange(1+i*len(x_pred), (i+1)*len(x_pred)+1)
            index.shape = (len(x_pred),1)          
            data_meta[i*len(x_pred):(i+1)*len(x_pred), :] = npy.concatenate((index, a[i]*npy.ones((len(x_pred),1)), y_pred), axis = 1)
      
    data = data_meta[data_meta[:,2] > min(data_meta[:,2])]
    
    return data, statistics