import numpy as np

###
### Calculate the Kullback-Leiber Divergence
###
def dKL(p,q,axis=1):
    ##Protect from np.log2(0)'s
    q[q==0]=1e-16
    p[p==0]=1e-16    
    return np.sum(p*np.log2(p)-p*np.log2(q),axis=axis)

###
### Calculate the Jensen-Shannon Divergence
###
def dJS(p,q,axis=1):
    m=0.5*(q+p)
    return 0.5*dKL(p,m,axis=axis)+0.5*dKL(q,m,axis=axis)
