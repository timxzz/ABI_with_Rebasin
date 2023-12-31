""" MNIST HMC
    Ref: https://github.com/AdamCobb/hamiltorch/tree/master/notebooks
"""
# %%
import numpy as np

import torch
from torchvision import datasets, transforms

import copy

import utils
from MNIST_models import MNIST_Ensemble, to_numpy
import MNIST_VI
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

    ## Manually transforms.ToTensor() for full batch HMC
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False)

    num_samples=100

    # Get HMC predictive distributions
    pred_dist_hmc = get_HMC_pred_dists(run_name_hmc, device, test_loader, N_ensemble=1000)

    # Get Ensemble predictive distributions
    pred_dist_ens, pred_dist_rb_m, pred_dist_rb_svd, pred_dist_rbmf = get_Ensemble_pred_dists(run_name_ens, 
                                                                                        device, 
                                                                                        test_loader, 
                                                                                        num_samples=num_samples,
                                                                                        train_loader=train_loader,
                                                                                        )

    # Get VI predictive distributions
    pred_dist_vi = get_VI_pred_dists(run_name_vi, device, test_loader, num_samples=num_samples)

    # Eval Agreement
    print("**** Agreement ****")
    ag_ens = metrics.agreement(predictions=pred_dist_ens, reference=pred_dist_hmc)
    ag_rb_m = metrics.agreement(predictions=pred_dist_rb_m, reference=pred_dist_hmc)
    ag_rb_svd = metrics.agreement(predictions=pred_dist_rb_svd, reference=pred_dist_hmc)
    ag_rbmf = metrics.agreement(predictions=pred_dist_rbmf, reference=pred_dist_hmc)
    ag_vi = metrics.agreement(predictions=pred_dist_vi, reference=pred_dist_hmc)

    print(f"Ensemble Agreement: {ag_ens}")
    print(f"Ensemble Rebased Mean Agreement: {ag_rb_m}")
    print(f"Ensemble Rebased SVD Agreement: {ag_rb_svd}")
    print(f"Ensemble Rebased Mean Field Agreement: {ag_rbmf}")
    print(f"VI Agreement: {ag_vi}")

    # Eval TV
    print("**** TV ****")
    tv_ens = metrics.total_variation_distance(predictions=pred_dist_ens, reference=pred_dist_hmc)
    tv_rb_m = metrics.total_variation_distance(predictions=pred_dist_rb_m, reference=pred_dist_hmc)
    tv_rb_svd = metrics.total_variation_distance(predictions=pred_dist_rb_svd, reference=pred_dist_hmc)
    tv_rbmf = metrics.total_variation_distance(predictions=pred_dist_rbmf, reference=pred_dist_hmc)
    tv_vi = metrics.total_variation_distance(predictions=pred_dist_vi, reference=pred_dist_hmc)

    print(f"Ensemble TV: {tv_ens}")
    print(f"Ensemble Rebased Mean TV: {tv_rb_m}")
    print(f"Ensemble Rebased SVD TV: {tv_rb_svd}")
    print(f"Ensemble Rebased Mean Field TV: {tv_rbmf}")
    print(f"VI TV: {tv_vi}")




