""" MNIST HMC
    Ref: https://github.com/AdamCobb/hamiltorch/tree/master/notebooks
"""
# %%
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

import copy
import json

import utils
import hamiltorch
from MNIST_models import EnsNet, MNIST_Ensemble
from MNIST_Ensemble import evaluate


class Config:
    def __init__(self, **entries):
        for key, value in entries.items():
            if isinstance(value, dict):
                entries[key] = Config(**value)
        self.__dict__.update(entries)

def construct_config_string(args, hp_config):
    """
    Construct a configuration string based on the attributes of args and hp_config.
    - args: An object with attributes representing primary configuration parameters.
    - hp_config: A nested config object with attributes representing hyperparameters.
    """
    def get_attr_string(obj):
        """Helper function to get attribute key-value pairs as strings."""
        fragments = []
        for key, value in vars(obj).items():
            # If the value is an object, recursively get its attributes
            if hasattr(value, "__dict__"):
                fragments.extend(get_attr_string(value))
            else:
                fragments.append(f"-{key}_{value}")
        return fragments

    config_string = "HMC" + ''.join(get_attr_string(args) + get_attr_string(hp_config))
    
    return config_string

def object_to_flatten_dict(obj, parent_key='', sep='.'):
    items = {}
    for k, v in vars(obj).items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if hasattr(v, "__dict__"):
            items.update(object_to_flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

# load hyperparameters from HP_hmc.json
def load_hmc_hyperparameters(dataset_name, config_name='test'):
    with open('HP_hmc.json', 'r') as f:
        HP_dict = json.load(f)
    # assert that the dataset_name and config_name are valid
    assert dataset_name in HP_dict.keys(), "dataset_name not found in HP_dict"
    assert config_name in HP_dict[dataset_name].keys(), "config_name not found in HP_dict[dataset_name]"

    # Convert the dictionary to an object
    config = Config(**HP_dict[dataset_name][config_name])
    return config

# format the kwargs from config
def get_net_kwargs(config, device):
    kwargs = {
            'input_dim':config.input_dim, 
            'output_dim':config.output_dim, 
            'init_std': config.init_std,
            'device': device}
    return kwargs

# plot inline samples eval
def plot_inline_sample_acc_ll(acc, nll):
    fs = 20
    plt.figure(figsize=(10,5))
    plt.plot(acc)
    plt.grid()
    # plt.xlim(0,3000)
    plt.xlabel('Iteration number',fontsize=fs)
    plt.ylabel('Sample accuracy',fontsize=fs)
    plt.tick_params(labelsize=15)
    # plt.savefig('mnist_acc_100_training.png')
    plt.show()

    fs = 20
    plt.figure(figsize=(10,5))
    plt.plot(nll)
    plt.grid()
    # plt.xlim(0,3000)
    plt.xlabel('Iteration number',fontsize=fs)
    plt.ylabel('Negative Log Likelihood',fontsize=fs)
    plt.tick_params(labelsize=15)
    # plt.savefig('mnist_acc_100_training.png')
    plt.show()

# %%

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)

    # Check if in an interactive environment (For dev)
    import sys
    if hasattr(sys, 'ps1') or 'IPYTHON' in sys.modules:
        print("In an interactive environment")
        sys.argv = ['']  # Clear command-line arguments

    # convert command line arguments to dictionary
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')

    parser.add_argument(
        "--dataset", choices=["MNIST"], default="MNIST",
        help="Choose the dataset for the experment.")
    parser.add_argument(
        "--config_name", default="test",
        help="Specified the hyperparameter config group to use.")

    parser.add_argument(
        "--run_name", default="",
        help="Specified the name of the run.")
    parser.add_argument(
        "--run_batch_name", default="test",
        help="Specified the name of the batch for runs if doing a batch grid search etc.")

    args = parser.parse_args()

    # load hyperparameters from HP_hmc.json
    hp_config = load_hmc_hyperparameters(args.dataset, args.config_name)


    config_string = construct_config_string(args, hp_config)
    print("configs: ", config_string)

    if args.run_name == "":
        run_name = "HMC-" + args.dataset +"-"+ args.config_name
    else:
        run_name = "HMC-" + args.dataset +"-"+ args.config_name +"-"+ args.run_name
    print("run_name: ", run_name)

    # setting up the MNIST dataset
    transform=transforms.Compose([
            transforms.ToTensor(),])
    train_data = datasets.MNIST('./data', train=True, download=False,
                        transform=transform)
    test_data = datasets.MNIST('./data', train=False, download=False,
                        transform=transform)

    # Manually transforms.ToTensor() for full batch HMC
    x_train = train_data.data.float()/255.
    x_train = x_train[:,None,:,:].to(device)
    y_train = train_data.targets.reshape((-1,1)).float().to(device)
    x_test = test_data.data.float()/255.
    x_test = x_test[:,None,:,:].to(device)
    y_test = test_data.targets.reshape((-1,1)).float().to(device)


    kwargs = get_net_kwargs(hp_config.net, device)
    hmc_config = hp_config.hmc

    net = EnsNet(**kwargs)

    # Hamiltorch parameters
    ## prior percision
    tau_list = []
    # tau = 1. #1. #10.#./100. # 1/50
    for w in net.parameters():
        # print(w.nelement())
        # tau_list.append(tau/w.nelement())
        tau_list.append(hmc_config.tau)
    tau_list = torch.tensor(tau_list).to(device)


    ## HMC
    hamiltorch.set_random_seed(hmc_config.seed)
    params_init = hamiltorch.util.flatten(net).to(device).clone()
    # print num of params
    print("Shape of parameters: ", params_init.shape)

    params_hmc = hamiltorch.sample_model(net, 
                    x_train, y_train, 
                    params_init=params_init, 
                    model_loss='multi_class_linear_output', 
                    num_samples=        hmc_config.num_samples, 
                    burn =              hmc_config.burn,
                    step_size=          hmc_config.step_size, 
                    num_steps_per_sample=   hmc_config.L,
                    tau_out=            hmc_config.tau_out, 
                    tau_list=tau_list, 
                    store_on_GPU=       False,
                    normalizing_const=  hmc_config.normalizing_const)
    
    # %%
    pred_list, log_prob_list = hamiltorch.predict_model(net, 
                    x = x_test, y = y_test, 
                    samples=params_hmc, 
                    # model_loss='multi_class_log_softmax_output', 
                    model_loss='multi_class_linear_output', 
                    tau_out=1., 
                    tau_list=tau_list,
                    device=device)

    pred_prob_list = F.softmax(pred_list, dim=-1)

    acc = torch.zeros( int(len(params_hmc)))
    acc_test_list = torch.zeros( int(len(params_hmc)))
    nll = torch.zeros( int(len(params_hmc)))
    ensemble_proba = F.softmax(pred_list[0].to(device), dim=-1)
    for s in range(0,len(params_hmc)):
        _, pred = torch.max(pred_list[:s].mean(0), -1)
        _, pred_single = torch.max(pred_list[s].to(device), -1)
        acc[s] = (pred.to(device).float() == y_test.flatten()).sum().float()/y_test.shape[0]
        acc_test_list[s] = (pred_single.float() == y_test.flatten()).sum().float()/y_test.shape[0]
        ensemble_proba += F.softmax(pred_list[s].to(device), dim=-1)
        nll[s] = F.nll_loss(torch.log(ensemble_proba.cpu()/(s+1)), y_test[:].long().cpu().flatten(), reduction='mean')
    # %%
    # plot_inline_sample_acc_ll(acc, nll)
    print("Accuracy: ", acc[-1])
    print("NLL: ", nll[-1])


    # %%
    print("Number of samples (including params_init): ", len(params_hmc))
    print("Shape of a sample: ", params_hmc[0].shape)

    _hmc_samples = {
        'params': params_hmc,
        'pred_prob_list': pred_prob_list,
        'acc_test_list': acc_test_list
    }

    # save HMC samples 'params_hmc' 
    print("**** Save HMC samples ****")
    utils.save_hmc_samples(_hmc_samples, kwargs, run_name)

    print("**** Load HMC samples ****")
    hmc_samples, kwargs = utils.load_hmc_samples(run_name)

    # extent kwargs to ensemble kwargs
    ens_kwargs = copy.deepcopy(kwargs)
    # ens_kwargs['N_ensemble'] = len(params_hmc)
    ens_kwargs['N_ensemble'] = 10

    # create the ensemble
    ens = MNIST_Ensemble(**ens_kwargs)
    ens.set_weights(hmc_samples['params'][-10:])



    print("**** Test ensemble ****")
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False)
    accuracy = evaluate(ens, test_loader, device)
    print('Test Accuracy: {}%'.format(accuracy))

