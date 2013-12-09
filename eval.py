### PyMLSEC Toolbox
# Performance evaluation
# Copyright (c) 2013 Konrad Rieck <konrad@mlsec.org>


""" This module provides functions for evaluating the performance of 
classification and detection methods. 
"""

import sys
import math
import numpy as np
import matplotlib.pyplot as plt


def average_roc(rocs, fps):    
    """Average a list of rocs over a grid."""
    fps = filter(lambda f: f < 1, fps)    
    tps = np.zeros((len(rocs), len(fps)))
    
    # loop over ROCs
    for i in range(len(rocs)):
        
        # interpolate tps over fp grid
        r = rocs[i]        
        for j in range(len(fps)):
            k = np.min(np.nonzero(r[1,:] > fps[j]))
            s = (r[1, k+0] - fps[j]) / (r[1, k+0] - r[1, k-1])
            tps[i,j] = s * (r[0, k+0] - r[0, k-1]) + r[0, k-1]

    # return average ROC and std
    avg_roc = np.vstack((np.mean(tps, 0), fps))
    std_roc = np.vstack((np.std(tps, 0), fps))
    return (avg_roc, std_roc)
        
def plot_average_roc(avg_roc, std_roc, boundary, filename):

    std0 = std_roc[1]
    std1 = avg_roc[0] + std_roc[0]
    std2 = avg_roc[0] - std_roc[0]
    plt.figure(figsize=(20,18))
    plt.plot(avg_roc[1], avg_roc[0], 'k-', std0, std1, 'k--', std0, std2, 'k--')
    plt.legend(('Average ROC', 'StdDev ROC'), 'lower right', shadow=True)
    plt.xlabel('False Positive Rate')
    plt.xlim((0.0, boundary))
    plt.ylabel('True Positive Rate')
    plt.ylim((0.0, 1.0))
    plt.title("Average ROC")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig( filename, format='png')

def compress_roc(roc):
    """Compress ROC curve by removing irrelavent points."""
    comp = np.array([[0,0]])    
    
    # loop over ROC
    for i in range(1, roc.shape[1]):        
        # add current and previous points if direction changes
        if (comp[-1,:] == roc[:,i]).sum() == 0:
            comp = np.vstack((comp, roc[:,i - 1]))
            comp = np.vstack((comp, roc[:,i]))
    comp = np.vstack((comp, [1,1]))

    return np.transpose(comp)

def compute_roc(s, y):
    """Compute ROC curve from scores and labels (y <= 0 or > 0)."""
   
    # Weird things happen for non-numpy array
    assert isinstance(s, np.ndarray)
    assert isinstance(y, np.ndarray)

    # change type to nunpy.float32
    s = np.float32(s)
    y = np.float32(y)

    # sort by scores
    idx = np.argsort(s)
    s = s[idx]
    y = y[idx]

    # set labels to 0 and 1
    y[np.nonzero(y >  0)] = 1.0
    y[np.nonzero(y <= 0)] = 0.0
    
    # initialize 
    truth = np.array([y.sum(), len(y) - y.sum()])
    preds = np.array([0, 0])
    roc = np.zeros((2, len(s) + 1))

    # compute ROC curve
    for i in range(len(s)):
        roc[:,i] = preds / truth
        if y[i] > 0:
            preds[0] += 1
        else:
            preds[1] += 1
        roc[:,-1] = [1,1]
    roc = np.fliplr(1 - roc)
            
    # compress ROC curve
    # roc = compress_roc(roc)            
    return roc

def compute_auc(roc, b = 1):
    """Compute AUC value from ROC curve (for bounded fp-rate)"""
    r = np.copy(roc)
    idx = np.nonzero(r[1,:] < b)[0]

    a = 1    
    # bound AUC at b
    if b < 1:
        j = np.max(idx)
        m = (r[0,j+1] - r[0,j]) / (r[1,j+1] - r[1,j])
        n = r[0,j+1]
        a = b * m + n

    r = r[:,idx]
    r = np.append(r, [[a],[b]], 1)

    auc = (np.diff(r[1,:]) * r[0,:-1]).sum()
    auc = auc / b
    return auc
    
