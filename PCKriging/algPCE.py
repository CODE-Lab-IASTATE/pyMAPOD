# Filename: algorithms.py

# Created: July/02/2018
# Last modified: July/07/2018
# Author: Prof. Leifur Leifsson
#         PhD student: Xiaosong    

import numpy as npy
from scipy import stats
from scipy import integrate
from numpy import linalg as LA

def gen_quad(n_deg, key, value):
    
    '''inputs:
       n_deg: required degree of PCE, scalar
       key: distribution type of random variable, string
       value: parameters of random distribution, tuple
       outputs:
       weight: Gaussian quadrature weighting factors
       x_quad: Gaussian quadrature points
       alpha: coefficients of generalized recurrent relationship
       beta: coefficients of generalized recurrent relationship
    '''
    
    # This function aims at obtaining alpha and beta
    # in turn to generate Gaussian qudarture points and weighting factors
    # therefore we can use Eqn. 1.1 [Walter Gaustschi, Matlab software]    
    alpha = npy.zeros((n_deg + 1, 1))
    beta  = npy.zeros((n_deg + 1, 1))
    Jac   = npy.zeros((n_deg + 1, n_deg + 1))

    for i in xrange(0, n_deg + 1):

        if i == 0:
            pi_old = lambda x: 0
            pi_new = lambda x: 1
            # here use pi_new and pi_new to make beta[0] = 1
            # due to the setup of beta = integral(f*dx), Eqn 1.2, Walter(2004)
            [alpha[i], beta[i]] = Chris_Darb(pi_new, pi_new, key, value)
            Jac[i,i] = alpha[i]
        else:
            # pi_gen = lambda x: (x - alpha[i-1])*pi_new(x) - beta[i-1]*pi_old(x)
            # define the recursion of pi_gen in a function, instead of a single line
            # because it will give the error called "maximum recursion depth exceeded"
            pi_gen = rec_func(alpha[i-1], beta[i-1], pi_new, pi_old)
            pi_old = pi_new
            pi_new = pi_gen
            [alpha[i], beta[i]] = Chris_Darb(pi_new, pi_old, key, value)
            Jac[i, i-1] = npy.sqrt(beta[i])
            Jac[i-1, i] = npy.sqrt(beta[i])
            Jac[i, i]   = alpha[i]
            
    [x_quad, v_quad] = LA.eig(Jac)
    weight = (v_quad[0, :])**2
    
    return alpha, beta, x_quad, weight
    

def rec_func(alpha, beta, pi_new, pi_old):
    
    pi_gen = lambda x: (x - alpha)*pi_new(x) - beta*pi_old(x)
    
    return pi_gen

def Chris_Darb(pi_new, pi_old, key, value):
    
    '''inputs:
       f1: current basis of x, function string
       f2: previous basis of x, function string
       key: distribution type of random variable, string
       value: parameters of random distribution, tuple
       outputs:
       alpha: coefficients of generalized recurrent relationship
       beta: coefficients of generalized recurrent relationship
    '''
    
    f = lambda x: x * pi_new(x)
    alpha = adaptive_integrate(f, pi_new, key, value) / adaptive_integrate(pi_new, pi_new, key, value)
    beta = adaptive_integrate(pi_new, pi_new, key, value) / adaptive_integrate(pi_old, pi_old, key, value)
    
    return alpha, beta
    

def adaptive_integrate(f1, f2, key, value):

    '''inputs:
       f1: function 1 of x, function string
       f2: function 2 of x, function string
       key: distribution type of random variable, string
       value: parameters of random distribution, tuple
       outputs:
       y: integral value
    '''
    
    if key.startswith('Uniform'):
        # stats.uniform defined in the range of [0, 1]
        # we have to convert it to [-1, 1] for the definition of Legendre basis
        # stats.uniform(location, scale)
        # or we can also do arbitrary type, will work on this later
        f_distr = stats.uniform(-1, 2)
        f0 = lambda x: f_distr.pdf(x)
        f = lambda x: f1(x) * f2(x) * f0(x)
        y = integrate.quad(f, -1, 1)
        
    elif key.startswith('Gaussian'):
        # this is for hermite polynomial basis
        # we can do arbitrary type by not using standard normal distribution
        # will work on this later
        f_distr = stats.norm(0, 1)
        f0 = lambda x: f_distr.pdf(x)
        f = lambda x: f1(x) * f2(x) * f0(x)
        y = integrate.quad(f, -npy.inf, npy.inf)
        
    elif key.startswith('Gamma'):
        # compare the stats.gamma with the one showed in UQLab tutorial (input)
        # stats.gamma accepts only one value, but UQLab accepts two
        # we can do the location and scale to make them the same
        # argument "1" is for the "standardized" format
        # or we can do arbitrary type later
        # value[0]: lambda, value[1]: k (a for stats.gamma)
        a = value[1]
        loc = 0
        scale = 1./value[0] # stats.gamma uses "beta" instead of "lambda"
        f_distr = stats.gamma(a, loc, scale)
        f0 = lambda x: f_distr.pdf(x)
        f = lambda x: f1(x) * f2(x) * f0(x)
        y = integrate.quad(f, 0, npy.inf)
        
    elif key.startswith('Beta'):
        # compare the stats.beta with the one showed in UQLab tutorial (input)
        # stats.beta accepts only one value, but UQLab accepts two
        # we can do the location and scale to make them the same
        # value[0]: alpha, value[1]: beta, no "loc" or "scale" needed
        # always in the range of [0, 1]
        alpha = value[0]
        beta = value[1]
        f_distr = stats.beta(alpha, beta)
        f0 = lambda x: f_distr.pdf(x)
        f = lambda x: f1(x) * f2(x) * f0(x)
        y = integrate.quad(f, 0, 1)
    
    elif key.startswith('Exponential'):
        # value: lambda
        loc = 0
        scale = 1./value
        f_distr = stats.expon(loc, scale)
        f0 = lambda x: f_distr.pdf(x)
        f = lambda x: f1(x) * f2(x) * f0(x)
        y = integrate.quad(f, 0, npy.inf)
    
    elif key.startswith('Lognormal'):
        # this part is very interesting
        # in UQLab they do Hermite for lognormal
        # and U the same as those from gaussian
        # then convert U to X using exp(U)
        # or they can specify arbitrary polynomial basis to be the same as here
        # we can do both, actually
        
        # value[0]: mu, value[1]:sigma
        s = value[1]
        loc = 0
        scale = npy.exp(value[0])
        f_distr = stats.lognorm(s, loc, scale)
        f0 = lambda x: f_distr.pdf(x)
        f = lambda x: f1(x) * f2(x) * f0(x)
        y = integrate.quad(f, 0, npy.inf)
    
    elif key.startswith('Gumbel'):
        # compare the stats.gumbel_r with the one showed in UQLab tutorial (input)
        # stats.gamma accepts only one value, but UQLab accepts two
        # we can do the location and scale to make them the same
        # value[0]: mu, value[1]: beta
        loc = value[0]
        scale = value[1]
        f_distr = stats.gumbel_r(loc, scale)
        f0 = lambda x: f_distr.pdf(x)
        f = lambda x: f1(x) * f2(x) * f0(x)
        y = integrate.quad(f, -npy.inf, npy.inf)
        
    elif key.startswith('Weibull'):
        # compare the stats.weibull_min with the one showed in UQLab tutorial (input)
        # stats.gamma accepts only one value, but UQLab accepts two
        # we can do the location and scale to make them the same
        # value[0]: lambda, value[1]: k
        k  = value[1]
        loc = 0
        scale = value[0]
        f_distr = stats.weibull_min(k, loc, scale)
        f0 = lambda x: f_distr.pdf(x)
        f = lambda x: f1(x) * f2(x) * f0(x)
        y = integrate.quad(f, 0, npy.inf)
        
    elif key.startswith('Triangular'):
        # compare the stats.triang with the one showed in UQLab tutorial (input)
        # stats.gamma accepts only one value, but UQLab accepts two
        # we can do the location and scale to make them the same
        # value: c, no "loc" and "scale" needed
        # always in the range of [0, 1]
        c = value
        f_distr = stats.triang(c)
        f0 = lambda x: f_distr.pdf(x)
        f = lambda x: f1(x) * f2(x) * f0(x)
        y = integrate.quad(f, 0, 1)
        
    elif key.startswith('Logistic'):
        # compare the stats.logistic with the one showed in UQLab tutorial (input)
        # stats.gamma accepts only one value, but UQLab accepts two
        # we can do the location and scale to make them the same
        # value[0]: location, value[1]: scale
        loc = value[0]
        scale = value[1]
        f_distr = stats.logistic(loc, scale)
        f0 = lambda x: f_distr.pdf(x)
        f = lambda x: f1(x) * f2(x) * f0(x)
        y = integrate.quad(f, -npy.inf, npy.inf)

    elif key.startswith('Laplace'):
        # compare the stats.laplace with the one showed in UQLab tutorial (input)
        # stats.gamma accepts only one value, but UQLab accepts two
        # we can do the location and scale to make them the same
        # value[0]: location, value[1]: scale
        loc = value[0]
        scale = value[1]
        f_distr = stats.laplace(loc, scale)
        f0 = lambda x: f_distr.pdf(x)
        f = lambda x: f1(x) * f2(x) * f0(x)
        y = integrate.quad(f, -npy.inf, npy.inf)
                
    else:
        print 'other types of statistical distributsions are coming soon ...'

    return y[0]
    
  
def gen_basis(n_deg, key, value, alpha, beta, x_quad, weight):
    
    '''inputs:
       n_deg: required degree of PCE, scalar
       key: distribution type of random variable, string
       value: parameter of the distribution, tuple
       alpha: reccurence coefficient
       beta:  reccurence coefficient
       x_quad: Gauss quadrature nodes
       weight: Gauss quadrature weights
       outputs:
       basis: basis of PCE, npy.ndarray
    '''
    
    basis = npy.ones((len(x_quad), n_deg+1))
    
    integration = npy.ones((len(x_quad), n_deg+1))
    
    pi_old = lambda x: 0
    pi_new = lambda x: 1
    
    for i in xrange(1, n_deg + 1):

        pi_gen = rec_func(alpha[i-1], beta[i-1], pi_new, pi_old)
        pi_old = pi_new
        pi_new = pi_gen
        
        integration[:, i] = npy.sqrt(adaptive_integrate(pi_new, pi_new, key, value))
        
        basis[:, i] = pi_new(x_quad) / integration[:, i]
        
    return basis, integration     
    
       
def convert_x(x_quad, key, value):
    
    if key.startswith('Uniform'):
        x_quad = (value[1] - value[0]) / 2. * x_quad + (value[1] + value[0]) / 2.
    elif key.startswith('Gaussian'):
        x_quad = value[0] + value[1] * x_quad
    else:
        pass
    
    return x_quad
    

def convert_x_inv(x_exp, key, value):
    
    if key.startswith('Uniform'):
        x_exp = (2. * x_exp - (value[1] + value[0])) / (value[1] - value[0])
    elif key.startswith('Gaussian'):
        x_exp = (x_exp - value[0]) / value[1]
    else:
        pass
    
    return x_exp