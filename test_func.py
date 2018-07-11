import numpy as npy

def simpleFunc(a, para):
    
    k = para[0]
    b = para[1]
    
    y = npy.exp(k*npy.log(a) + b)
    
    return y