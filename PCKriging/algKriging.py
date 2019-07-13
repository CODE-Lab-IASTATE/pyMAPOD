# algKriging

# created on July 27, 2018
# modified on July 27, 2018
# developed by Leifur Leifsson (assisted professor)
#              Xiaosong Du (Ph.D. student)

import numpy as npy

def infMatrix(setting, *args):
    
    if setting['trendType'] == 'simple':
        print 'under construction'
        
    elif setting['trendType'] == 'ordinary':
        
        F = npy.ones([len(setting['exp_x']), 1])
        
    elif setting['trendType'] == 'linear':
        
        F = npy.ones([setting['exp_x'].shape[0], setting['exp_x'].shape[1] + 1])
        
        for i in xrange(0, setting['exp_x'].shape[1]):
            F[:, i+1] = setting['exp_x'][:, i]
            
    elif setting['trendType'] == 'pod':
        
        F = npy.ones([setting['exp_x'].shape[0], 2])
        
        F[:, 1] = setting['exp_x'][:, 0]
        
    elif setting['trendType'] == 'quadratic':
        print 'under construction'
        
    elif setting['trendType'] == 'polynomial':
        # need to redefine the dimensionality of F
        # based on the user-defined degree
        print 'under construction'
    
    elif setting['trendType'] == 'pce':
        PCE = args[0]
        F = PCE['basis_coef_total'][:, PCE['coef_index'],0]
            
    else:
        print 'no such type of trend function'
        
    return F
    
def corMatrix(setting):
    
    # here will avoid to use matrix of functions
    # so, there will be no term related to theta
    # will define a 3d matrix
    # the 3rd dimensionality will be divided by theta in optimizer
    
    # get the number of training points, and dimensionality of random inputs
    [N, M] = npy.shape(setting['exp_x'])
    
    # define the 3d matrix explained above
    R = npy.zeros([N, N, M])
    
    # define the vector h, which will be used as the 3rd dimensionality
    h = npy.zeros([M, 1])
    
    for i in xrange(0, N):
        for j in xrange(0, i):
            h = npy.abs(setting['exp_x'][i, :] - setting['exp_x'][j, :])
            R[i, j, :] = h
            R[j, i, :] = h
            
    return R
    
def thetaFunc(theta, F, R, setting):
    
    R_pred = RFunc(theta, R, setting)
    
    beta = betaFunc(F, R_pred, setting)
    
    var = varFunc(beta, F, R_pred, setting)
    
    f_obj = 1/2. * npy.linalg.slogdet(R_pred)[1] \
          + len(setting['exp_y'])/2. * npy.log(2 * npy.pi * var) \
          + len(setting['exp_y'])/2.
    
    return f_obj
    
def RFunc(theta, R, setting):
    
    [N, M] = npy.shape(setting['exp_x'])

    R_pred = npy.ones([N, N]) 
    
    if setting['corrType'].startswith('ellipsoidal'):
        for i in xrange(0, N):
            for j in xrange(0, N):
                R_pred[i, j] = npy.linalg.norm(R[i, j, :] / theta)
  
        R_pred = corrFam(R_pred, setting)
            
    elif setting['corrType'].startswith('separable'):
        R_temp = npy.ones([N, N, M])
        for i in xrange(0, N):
            for j in xrange(0, N):
                R_temp[i, j, :] = corrFam(R[i, j, :] / theta, setting)
                
                R_pred[i, j] = npy.prod(R_temp[i, j, :])
                
    else:
        print 'no such type of correlation function'
    
    npy.fill_diagonal(R_pred, npy.diagonal(R_pred) + setting['nugget'])
       
    return R_pred
    
def corrFam(R, setting):
    
    if setting['corrFam'] == 'linear':
        R_temp = npy.maximum(0, 1 - R)
        
    elif setting['corrFam'] == 'exponential':
        R_temp = npy.exp(-R)
        
    elif setting['corrFam'] == 'gaussian':
        R_temp = npy.exp(-(R * R) / 2.)
        
    elif setting['corrFam'] == 'matern_3_2':
        R_temp = (1 + npy.sqrt(3.) * R) * npy.exp(-npy.sqrt(3.) * R)
        
    elif setting['corrFam'] == 'matern_5_2':
        R_temp = (1 + npy.sqrt(5.) * R + 5. / 3. * R * R) * npy.exp(-npy.sqrt(5.) * R)
        
    else:
        print 'no such correlation family'
        
    return R_temp
        

def betaFunc(F, R_pred, setting):
    
    beta = npy.matmul(npy.transpose(F), npy.linalg.inv(R_pred)) 
       
    beta = npy.matmul(beta, F)   
    
    beta = npy.matmul(npy.linalg.inv(beta), npy.transpose(F))
    
    beta = npy.matmul(beta, npy.linalg.inv(R_pred))
    
    beta = npy.matmul(beta, setting['exp_y'])
    
    return beta
    

def varFunc(beta, F, R_pred, setting):
    
    mat = setting['exp_y'] - npy.matmul(F, beta)

    var = npy.matmul(npy.transpose(mat), npy.linalg.inv(R_pred))
    
    var = npy.matmul(var, mat)
    
    var = var / len(setting['exp_y'])
    
    return var
    
def infMatrix_pred(x_pred, setting, *args):
    
    if setting['trendType'] == 'simple':
        
        print 'under construction'
        
    elif setting['trendType'] == 'ordinary':
        
        f = npy.ones([1, len(x_pred)])
        
    elif setting['trendType'] == 'linear':
        
        f = npy.ones([x_pred.shape[1] + 1, x_pred.shape[0]])
        
        for i in xrange(0, x_pred.shape[1]):
            f[i+1, :] = x_pred[:, i]
            
    elif setting['trendType'] == 'pod':
        
        f = npy.ones([2, x_pred.shape[0]])
        
        f[1, :] = x_pred[:, 0]
        
    elif setting['trendType'] == 'quadratic':
        
        print 'under construction'
        
    elif setting['trendType'] == 'polynomial':
        
        # need to redefine the dimensionality of F
        # based on the user-defined degree
        print 'under construction'
        
    elif setting['trendType'] == 'pce':
        
        PCE = args[0]
        PCE_pred = args[1]
        f = npy.transpose(PCE_pred['basis_total'][:, PCE['coef_index']])
        
    else:
        
        print 'no such type of trend function'
        
    return f
    
def corMatrix_pred(x_pred, setting):
    
    # here will avoid to use matrix of functions
    # so, there will be no term related to theta
    # will define a 3d matrix
    # the 3rd dimensionality will be divided by theta in optimizer
    
    # get the number of training points, and dimensionality of random inputs
    [N, M] = npy.shape(x_pred)
    
    # define the 3d matrix explained above
    R = npy.zeros([N, len(setting['exp_x']), M])
    
    # define the vector h, which will be used as the 3rd dimensionality
    h = npy.zeros([M, 1])
    
    for i in xrange(0, N):
        for j in xrange(0, len(setting['exp_x'])):
            h = npy.abs(x_pred[i, :] - setting['exp_x'][j, :])
            R[i, j, :] = h
            
    return R
    
def RFunc_pred(theta, r, x_pred, setting):
    
    [N, M] = npy.shape(x_pred)

    r_pred = npy.ones([N, len(setting['exp_x'])]) 
    
    if setting['corrType'].startswith('ellipsoidal'):
        for i in xrange(0, N):
            for j in xrange(0, len(setting['exp_x'])):
                r_pred[i, j] = npy.linalg.norm(r[i, j, :] / theta)
                
        r_pred = corrFam(r_pred, setting)
            
    elif setting['corrType'].startswith('separable'):
        r_temp = npy.ones([N, len(setting['exp_x']), M])
        for i in xrange(0, N):
            for j in xrange(0, len(setting['exp_x'])):
                r_temp[i, j, :] = corrFam(r[i, j, :] / theta, setting)
                
                r_pred[i, j] = npy.prod(r_temp[i, j, :])
                
    else:
        print 'no such type of correlation function'
    
    npy.fill_diagonal(r_pred, npy.diagonal(r_pred) + setting['nugget'])
       
    return npy.transpose(r_pred)
