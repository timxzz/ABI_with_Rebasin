# %%
import numpy as np
import torch
from torchvision import datasets, transforms
import os
import copy
import json
from tqdm import tqdm

import utils
from MNIST_models import EnsNet, MNIST_Ensemble, to_numpy
import MNIST_VI
import MNIST_eval_pairplot
import metrics


# test the Sample Ensemble
def get_predictive_dist(model, dataloader, device):
    accuracy = 0
    predictive_dist = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device); y = y.to(device)
            # check if model is an MNIST_Ensemble object
            if isinstance(model, MNIST_Ensemble):
                # the "predict" function: by default dropout is on and we use K > 1 MC samples
                y_pred = model.predict_ens(x, reduce_mean=True)
            else:
                y_pred = model.predict(x)
            predictive_dist.append(y_pred)
            pred = y_pred.data.max(1, keepdim=True)[1] # get the index of the max probability
            accuracy += pred.eq(y.data.view_as(pred)).float().cpu().sum()
        accuracy = accuracy / len(dataloader.dataset) * 100 # accuracy in percentage
        predictive_dist = torch.cat(predictive_dist, dim=0)
    return to_numpy(accuracy), predictive_dist.cpu().numpy()


def get_HMC_pred_dists(run_name, device, test_loader, N_ensemble=10):
    print("**** Load HMC samples ****")
    hmc_samples, kwargs = utils.load_hmc_samples(run_name)
    kwargs['device'] = device

    # Evenly space integer from N to M with K samples
    selected_idxs = np.linspace(500, 9900-1, N_ensemble, dtype=int)
    if N_ensemble <= 20:
        print(f"Out of {len(hmc_samples['params'])} samples, use idx: {selected_idxs}")
    else:
        print(f"Out of {len(hmc_samples['params'])} samples, use {N_ensemble} evenly spaced samples")
        
    # extent kwargs to ensemble kwargs
    ens_kwargs = copy.deepcopy(kwargs)
    ens_kwargs['N_ensemble'] = N_ensemble

    # create the ensemble
    ens = MNIST_Ensemble(**ens_kwargs)
    selected_samples = [hmc_samples['params'][i] for i in selected_idxs]
    ens.set_weights(selected_samples)

    accuracy, pred_dist = get_predictive_dist(ens, test_loader, device)
    print('Test Accuracy: {}%'.format(accuracy))

    return pred_dist


def get_pred_dists_from_mean_std(mean, std, device, test_loader, kwargs,  N_ensemble=10):
    # mean and std to torch tensor
    if isinstance(mean, np.ndarray):
        mean = torch.tensor(mean, device=device)
        std = torch.tensor(std, device=device)

    kwargs['device'] = device
        
    # extent kwargs to ensemble kwargs
    ens_kwargs = copy.deepcopy(kwargs)
    ens_kwargs['N_ensemble'] = N_ensemble

    # create the ensemble
    ens = MNIST_Ensemble(**ens_kwargs)
    ens.nets = ens.sample_from_mf(mean, std, num_samples=N_ensemble)

    accuracy, pred_dist = get_predictive_dist(ens, test_loader, device)
    print('Test Accuracy: {}%'.format(accuracy))

    return pred_dist


def get_test_accs_from_pruned_mean_std(mean, std, device, percet_list, test_loader, kwargs,  net0=None):
    # mean and std to torch tensor
    if isinstance(mean, np.ndarray):
        mean = torch.tensor(mean, device=device)
        std = torch.tensor(std, device=device)

    kwargs['device'] = device

    # sort percet_list in descending order
    percet_list.sort(reverse=True)
    # Calculate number of samples for each percentage
    params_size = len(mean)
    num_samples = [int((perc/100) * params_size) for perc in percet_list]
    print(f"Number of samples: {num_samples}")
    # argsort from smallest to largest
    sorted_idx = np.argsort(std)
        

    accuracy_list = []
    for perc, n_sample in tqdm(zip(percet_list, num_samples), desc="Eval Pruned Mean", leave=False):
        if n_sample < params_size:
            n_pruned = params_size - n_sample
            zero_idx = sorted_idx[-n_pruned:]

            # create the mean net by setting the pruned weights to 0
            pruned_mean = mean.clone()
            pruned_mean[zero_idx] = 0.
        else:
            pruned_mean = mean.clone()
        mean_net = EnsNet(**kwargs)
        mean_net.set_weights(pruned_mean)
        accuracy, pred_dist = get_predictive_dist(mean_net, test_loader, device)
        accuracy_list.append(accuracy.item())

    print('Test Accuracy: {}%'.format(accuracy_list))

    net0_acc_list = []
    if net0 is not None:
        net0_weights = net0.get_flat_weights().clone()
        for perc, n_sample in tqdm(zip(percet_list, num_samples), desc="Eval Pruned Mean", leave=False):
            if n_sample < params_size:
                n_pruned = params_size - n_sample
                zero_idx = sorted_idx[-n_pruned:]

                # create the mean net by setting the pruned weights to 0
                pruned_mean = net0_weights.clone()
                pruned_mean[zero_idx] = 0.
            else:
                pruned_mean = net0_weights.clone()
            net0.set_weights(pruned_mean)
            accuracy, pred_dist = get_predictive_dist(net0, test_loader, device)
            net0_acc_list.append(accuracy.item())

        print('Net0 Test Accuracy: {}%'.format(net0_acc_list))


    return accuracy_list, net0_acc_list


def get_Ensemble_pred_dists(run_name, device, test_loader, num_samples=10, train_loader=None):
    print("**** Load Ensemble ****")
    ens = utils.load_ensemble(run_name, device=device)
    print(f"Ensemble size: {ens.N_ensemble}")

    print("**** Eval Ensemble ****")
    accuracy, pred_dist = get_predictive_dist(ens, test_loader, device)
    print('Test Accuracy: {}%'.format(accuracy))

    print("**** Eval Ensemble VRI ****")
    rb_net, rb_ws_m, rb_ws_std, rb_ws_B = ens.vri(train_loader, disable_rebasin=False)
    accuracy_rb_m, pred_dist_rb_m = get_predictive_dist(rb_net, test_loader, device)
    print('Rebased mean Test Accuracy: {}%'.format(accuracy_rb_m))


    nets = ens.sample_from_vri(rb_ws_m, rb_ws_B, num_samples=num_samples)
    samples_ens = copy.deepcopy(ens)
    samples_ens.N_ensemble = num_samples
    samples_ens.nets = nets
    accuracy_rb_svd, pred_dist_rb_svd = get_predictive_dist(samples_ens, test_loader, device)
    print(f'MC {num_samples} samples rebased svd Test Accuracy: {accuracy_rb_svd}%')

    nets = ens.sample_from_mf(rb_ws_m, rb_ws_std, num_samples=num_samples)
    samples_ens.nets = nets
    accuracy_rbmf, pred_dist_rbmf = get_predictive_dist(samples_ens, test_loader, device)
    print(f'MC {num_samples} samples rebased mean field Test Accuracy: {accuracy_rbmf}%')

    return pred_dist, pred_dist_rb_m, pred_dist_rb_svd, pred_dist_rbmf


def get_VI_pred_dists(run_name, device, test_loader, num_samples=10):
    print("**** Load VI ****")
    vi = utils.load_vi(run_name, device=device)

    accuracy, pred_dist = MNIST_VI.evaluate(vi, test_loader, device, K=num_samples)
    print(f'MC {num_samples} samples Test Accuracy: {accuracy}%')

    return pred_dist


# %%

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)

    # Check if in an interactive environment (For dev)
    import sys
    if hasattr(sys, 'ps1') or 'IPYTHON' in sys.modules:
        print("In an interactive environment")
        sys.argv = ['']  # Clear command-line arguments

    # run names
    run_name_hmc = "HMC-MNIST-test3.2-hmc_mnist_samples_10000_L500"

    run_name_ens = "test-Ensemble-N_ensemble_5-Ep_50-Adam-MAP_sigma_1.0"
    run_name_vi = "test_VI_epoch_50_Adam"


    # setting up the MNIST dataset
    transform=transforms.Compose([
            transforms.ToTensor(),])
    train_data = datasets.MNIST('./data', train=True, download=False,
                        transform=transform)
    test_data = datasets.MNIST('./data', train=False, download=False,
                        transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False)

    num_samples=200
    # without_rebasin = True
    without_rebasin = False

    # --------------------- Load the compact representation ---------------------

    kwargs = {"input_dim": 784, "output_dim": 10, "init_std": 0.05, "device": "cuda:0"}

    if os.path.exists("./saved_models/"+run_name_hmc+"/net0_hmc.pt") and not without_rebasin:
        print(f"**** Load HMC rebased mean and std from dir './saved_models/{run_name_hmc}' ****")
        # Load HMC rebased mean and std
        with open("./saved_models/"+run_name_hmc+ "/kwargs.json", "r") as f:
            kwargs = json.load(f)
        kwargs['device'] = device
        net0_hmc = EnsNet(**kwargs)
        net0_hmc.load_state_dict(torch.load("./saved_models/"+run_name_hmc+"/net0_hmc.pt"))
        m_hmc_std_hmc = np.load("./saved_models/"+run_name_hmc+"/m_hmc_std_hmc.npz")
        m_hmc = m_hmc_std_hmc['m_hmc']
        std_hmc = m_hmc_std_hmc['std_hmc']
    else:
        print(f"**** Get HMC rebased mean and std ****")
        # Get HMC rebased mean and std
        net0_hmc, m_hmc, std_hmc = MNIST_eval_pairplot.get_HMC_rebased_m_std(run_name_hmc, device, train_loader=train_loader, N_ensemble=1000, without_rebasin=without_rebasin)


    # Get Ensemble rebased mean and std
    net_neg1_hmc = MNIST_eval_pairplot.get_HMC_random_net(run_name_hmc, device, kwargs, N_ensemble=1000)
    m_ens, std_ens, net_neg1_ens= MNIST_eval_pairplot.get_Ensemble_net_neg1_rebased_m_std(run_name_ens, device, target_net=net0_hmc, train_loader=train_loader, without_rebasin=without_rebasin)

    # Get VI rebased mean and std
    m_vi, std_vi = MNIST_eval_pairplot.get_VI_rebased_m_std(run_name_vi, device, target_net=net0_hmc, train_loader=train_loader, without_rebasin=without_rebasin)

    # --------------------- Evaluate the Pruning ---------------------

    print("**** **** ****")
    print()
    print(f"**** Evaluate Pruning ****")

    percet_list = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 1]

    # Eval compact pruned mean
    print("**** Pruned HMC ****")
    accs_hmc = get_test_accs_from_pruned_mean_std(m_hmc, std_hmc, device, percet_list, test_loader, kwargs, net0=net_neg1_hmc)

    print("**** Pruned Ensemble ****")
    accs_ens = get_test_accs_from_pruned_mean_std(m_ens, std_ens, device, percet_list, test_loader, kwargs, net0=net_neg1_ens)

    print("**** Pruned Ensemble with HMC std ****")
    accs_ens = get_test_accs_from_pruned_mean_std(m_ens, std_hmc, device, percet_list, test_loader, kwargs, net0=net_neg1_ens)

    print("**** Pruned Ensemble with VI std ****")
    accs_ens = get_test_accs_from_pruned_mean_std(m_ens, std_vi, device, percet_list, test_loader, kwargs, net0=net_neg1_ens)

    print("**** Pruned VI ****")
    accs_vi = get_test_accs_from_pruned_mean_std(m_vi, std_vi, device, percet_list, test_loader, kwargs)



    # --------------------- Evaluate the representation ---------------------

    print("**** **** ****")
    print()
    print(f"**** Evaluate the representation ****")

    # Get HMC predictive distributions
    pred_dist_hmc = get_HMC_pred_dists(run_name_hmc, device, test_loader, N_ensemble=1000)

    # ******* Get Compact representation predictive distributions *******
    # Get HMC predictive distributions
    print("**** Sample from Cp.HMC ****")
    pred_dist_cp_hmc = get_pred_dists_from_mean_std(m_hmc, std_hmc, device, test_loader, kwargs,  N_ensemble=num_samples)

    # Get Ensemble predictive distributions
    print("**** Sample from Cp.Ensemble ****")
    pred_dist_cp_ens = get_pred_dists_from_mean_std(m_ens, std_ens, device, test_loader, kwargs,  N_ensemble=num_samples)
    
    # Get VI predictive distributions
    print("**** Sample from VI ****")
    pred_dist_vi = get_pred_dists_from_mean_std(m_vi, std_vi, device, test_loader, kwargs,  N_ensemble=num_samples)


    # Eval Agreement
    print("**** Agreement ****")
    ag_cp_hmc= metrics.agreement(predictions=pred_dist_cp_hmc, reference=pred_dist_hmc)
    ag_cp_ens = metrics.agreement(predictions=pred_dist_cp_ens, reference=pred_dist_hmc)
    ag_vi = metrics.agreement(predictions=pred_dist_vi, reference=pred_dist_hmc)

    print(f"Cp.HMC Agreement: {ag_cp_hmc}")
    print(f"Cp.Ensemble Agreement: {ag_cp_ens}")
    print(f"VI Agreement: {ag_vi}")

    # Eval TV
    print("**** TV ****")
    tv_cp_hmc = metrics.total_variation_distance(predictions=pred_dist_cp_hmc, reference=pred_dist_hmc)
    tv_cp_ens = metrics.total_variation_distance(predictions=pred_dist_cp_ens, reference=pred_dist_hmc)
    tv_vi = metrics.total_variation_distance(predictions=pred_dist_vi, reference=pred_dist_hmc)

    print(f"Cp.HMC TV: {tv_cp_hmc}")
    print(f"Cp.Ensemble TV: {tv_cp_ens}")
    print(f"VI TV: {tv_vi}")



