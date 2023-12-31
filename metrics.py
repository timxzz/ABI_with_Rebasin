'''
Ref: NeurIPS BDL Approximate Inference Competition - https://github.com/izmailovpavel/neurips_bdl_starter_kit 
Examples of organizer-provided metrics.
You can just replace this code by your own.
Make sure to indicate the name of the function that you chose as metric function
in the file metric.txt. E.g. example_metric, because this file may contain more 
than one function, hence you must specify the name of the function that is your metric.

'''
import torch
from torch import distributions as D
import numpy as np
import scipy

def agreement(predictions: np.array, reference: np.array):
    """Returns 1 if predictions match and 0 otherwise."""
    return (predictions.argmax(axis=-1) == reference.argmax(axis=-1)).mean()


def total_variation_distance(predictions: np.array, reference: np.array):
    """Returns total variation distance."""
    return np.abs(predictions - reference).sum(axis=-1).mean() / 2.


def w2_distance(predictions: np.array, reference: np.array):
    """Returns W-2 distance """
    NUM_SAMPLES_REQUIRED = 1000
    assert predictions.shape[0] == reference.shape[0], "wrong predictions shape"
    assert predictions.shape[1] == NUM_SAMPLES_REQUIRED, "wrong number of samples"
    return -np.mean([scipy.stats.wasserstein_distance(pred, ref) for 
                   pred, ref in zip(predictions, reference)])

def kl_a_b(a_m, a_std, b_m, b_std):
    """
    Compute KL(a||b) where a and b are normal distributions
    """
    # if input are numpy arrays, convert them to tensors
    if isinstance(a_m, np.ndarray):
        a_m = torch.tensor(a_m)
        a_std = torch.tensor(a_std)
    if isinstance(b_m, np.ndarray):
        b_m = torch.tensor(b_m)
        b_std = torch.tensor(b_std)
    pa = D.normal.Normal(a_m, a_std)
    
    pb = D.normal.Normal(b_m, b_std)

    # For: KL[p_a || p_b]
    kl = D.kl.kl_divergence(pa, pb)
    return kl
