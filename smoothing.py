import numpy as np
import scipy.stats as stats
from multiprocessing import Pool,cpu_count

class Smooth:
    width=1.0
    bins=50
    nx_d=500
    xlow=0.0
    xhigh=1.0
    cores=np.rint(2.*cpu_count()/3.)
    batch=20
    axis=0
    norm=True

    def batchSmooth(self,data):
        scale=1.
        if self.norm:
            if self.axis==0:
                scale=1./((data.T).shape[0])
            else:
                scale=1./(data.shape[0])
        pool = Pool(processes=self.cores)
        func=self.smfunc
        lists=pool.map(func,[data[j:j+self.batch] for j in range(0,data.shape[self.axis],self.batch)])
        pool.close()
        pool.join()
        return scale*np.concatenate([lists[i] for i in range(np.shape(lists)[self.axis])])
    
    def scipySmooth(self,data):
        x_d=np.linspace(self.xlow,self.xhigh,self.nx_d)
        frzNorm=stats.norm(data,self.width/self.bins)
        smooth=np.array([frzNorm.pdf(x_d[i]) for i in range(self.nx_d)])
        return np.sum(smooth,axis=2).T

    smfunc=scipySmooth
###
### Not terribly generic or safe code to do smoothing of a 2D array along one axis.  Smoothing
### is done in batches to keep the memory from exploding while still running quickly.  It
### is possible to direct the smfunc to a different smoothing function, but because pool is awful
### the alternative function must be a method of the class.
### 
### Usage:
### data = 2D array to be smoothed
### width = width of Gaussian normalized to number of bins
### bins = number of bins in histogram smooth distribution will be compared to.
### nx_d = over sampling for smooth distribution, for aesthetics should be >5x bins.
### axis = axis to slice across, the other is smeared.
### batch = Slices per thread to keep memory usage manageable. Best below 25, does not influence speed.
### cores = Number of cores to use.
### norm = Do normalization or not.
### xlow,xhigh = lower and upper bound of smearing range
###
### For broad features smooth with width ~ 1.5.  Narrow features 0.5.
###
### import smoothing
### sm=smoothing.Smooth()
### sm.cores=6
### sm.width=1.5
### pdf1=sm.batchSmooth(data1)
### sm.width=0.5
### pdf2=sm.batchSmooth(data2)

