from scipy.integrate import trapz
import numpy as np

def roc_curve(W10_pred, W1_gt, W0_gt, eps=1e-6, n_space=100):
    """
    W10_pred : W1-W0 predicted (unnormalized)
    W1_gt    : True adjacency matrix after perturbation
    W0_gt    : True adjacency matrix before perturbation

    Returns
    -----------


    Threshold used, True positive rate, False positive rate, Area under curve

    """
    # Normalize the prediction
    W10_pred = np.abs(W10_pred)
    W10_pred /= np.max(W10_pred)
    
    # Get W10_gt
    W10_gt = W0_gt - W1_gt
    
    threshold = np.linspace(-eps,1,n_space)
    
    tpr = np.zeros(threshold.shape)
    fpr = np.zeros(threshold.shape)
    
    W10_gt = W10_gt.flatten()
    W10_pred = W10_pred.flatten()

    i_keep = W0_gt.flatten()>0
    W10_gt  = W10_gt[i_keep]
    
    W10_pred = W10_pred[i_keep]
    
    i_gt = np.argwhere(W10_gt>0)
    tn_gt = np.argwhere(W10_gt<1)
    
    for i,t in enumerate(threshold):
        
        i_pred = np.argwhere(W10_pred>t)
        tp = len(np.intersect1d(i_pred, i_gt))
        fp = len(i_pred) - tp
        
        i_pred = np.argwhere(W10_pred<t)
        
        tn = len(np.intersect1d(i_pred, tn_gt))
        fn = len(i_pred) - tn
        
        if tp+fn>0:
            tpr[i] = tp/(tp+fn)
        if fp+tn>0:
            fpr[i] = fp/(fp+tn)
    
    aoc = trapz(tpr[::-1], fpr[::-1])
    return threshold, tpr, fpr,  aoc
    


def plot_roc_curve(grads, W1, W0, ax, title):
    
    aucs = np.zeros(len(grads))
    for i,grad in enumerate(grads):
        threshold, tpr, fpr, aucs[i] = roc_curve(grad, W1, W0)
        ax.plot(fpr, tpr, label="GCN Avr")

    # Xlim between 0 and 1
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.plot(xlim, ylim, "--", color="gray")
    
    eps = 0.05
    ax.set_xlim([0-eps, 1+eps]), ax.set_ylim([0-eps, 1+eps])

    ax.set_xticks(np.linspace(0,1,3))
    ax.set_yticks(np.linspace(0,1,3))
    ax.set_xlabel("False positive rate"), ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.text(0.99,0.01, r"AUC: %.1f%%"%(aucs[0]*100), ha="right", fontdict={"fontsize":12, "fontweight": "bold"})


    return aucs