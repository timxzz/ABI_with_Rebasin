import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
import random
import os
import json

from MNIST_models import MNIST_Ensemble, MNIST_VI

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def debug_memory():
    """ Debug memory usage """
    import collections, gc, resource, torch
    print('maxrss = {}'.format(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    tensors = collections.Counter(
        (str(o.device), str(o.dtype), tuple(o.shape))
        for o in gc.get_objects()
        if torch.is_tensor(o)
    )
    for line in sorted(tensors.items()):
        print('{}\t{}'.format(*line))

# save an ensemble to dir ./saved_models/save_name/
def save_ensemble(ens, kwargs, save_name):
    dir_name = "./saved_models/" + save_name + "/"
    # if dir does not exist, create it
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    # save kwargs in json file
    with open(dir_name + "kwargs.json", "w") as f:
        json.dump(kwargs, f)

    # save attributes
    torch.save(ens.__dict__, dir_name + "ens.pth")
    # save individual models
    for i in range(ens.N_ensemble):
        torch.save(ens.nets[i].state_dict(), dir_name + "net_" + str(i) + ".pth")

# load an ensemble from dir ./saved_models/save_name/
def load_ensemble(save_name, device=None):
    dir_name = "./saved_models/" + save_name + "/"
    # load kwargs from json file named kwargs.json
    with open(dir_name + "kwargs.json", "r") as f:
        kwargs = json.load(f)
    if device is not None:
        kwargs['device'] = device
    ens = MNIST_Ensemble(**kwargs)
    ens.__dict__.update(torch.load(dir_name + "ens.pth"))
    return ens

# save vi model to dir ./saved_models/save_name/
def save_vi(vi, kwargs, save_name):
    dir_name = "./saved_models/" + save_name + "/"
    # if dir does not exist, create it
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    # save kwargs in json file
    with open(dir_name + "kwargs.json", "w") as f:
        json.dump(kwargs, f)

    # save attributes
    torch.save(vi.__dict__, dir_name + "vi.pth")

# load vi model from dir ./saved_models/save_name/
def load_vi(save_name, device='cpu'):
    dir_name = "./saved_models/" + save_name + "/"
    # load kwargs from json file named kwargs.json
    with open(dir_name + "kwargs.json", "r") as f:
        kwargs = json.load(f)
    if device is not None:
        kwargs['device'] = device
    vi = MNIST_VI(**kwargs)
    vi.__dict__ = torch.load(dir_name + "vi.pth")
    return vi

# save hmc samples to dir ./saved_models/save_name/
def save_hmc_samples(hmc_samples, kwargs, save_name):
    dir_name = "./saved_models/" + save_name + "/"
    # if dir does not exist, create it
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    # save kwargs in json file
    with open(dir_name + "kwargs.json", "w") as f:
        json.dump(kwargs, f)

    # save attributes
    torch.save(hmc_samples, dir_name + "hmc_samples.pth")

# load hmc samples from dir ./saved_models/save_name/
def load_hmc_samples(save_name, device='cpu'):
    dir_name = "./saved_models/" + save_name + "/"
    # load kwargs from json file named kwargs.json
    with open(dir_name + "kwargs.json", "r") as f:
        kwargs = json.load(f)
    hmc_samples = torch.load(dir_name + "hmc_samples.pth", map_location=device)
    return hmc_samples, kwargs
