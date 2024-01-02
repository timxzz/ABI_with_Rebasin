# %%
import numpy as np

import torch
from torchvision import datasets, transforms

import copy

import utils
import rebasin
from MNIST_models import EnsNet, MNIST_Ensemble, to_numpy



def get_HMC_rebased_m_std(run_name, device, train_loader=None, N_ensemble=10, without_rebasin=False, rb2mean=False):
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

    if without_rebasin:
        # Get the mean and std from selected_samples
        ws_m, ws_std = ens.get_weights_m_std()
        return ens.nets[0], ws_m, ws_std

    if rb2mean:
        # Rebase the ensemble to the mean
        print("**** Rebase HMC to mean ****")
        m_net, _, _, _ = ens.vri(train_loader, disable_rebasin=True)
        _, rb_ws_m, rb_ws_std, _ = ens.vri(train_loader, target_net=m_net, disable_rebasin=False)
        return m_net, rb_ws_m, rb_ws_std
    else:
        # Rebase the ensemble to net0
        print("**** Rebase HMC ****")
        _, rb_ws_m, rb_ws_std, _ = ens.vri(train_loader, disable_rebasin=False)

        return ens.nets[0], rb_ws_m, rb_ws_std
    

def get_HMC_random_net(run_name, device, kwargs, N_ensemble=10):
    print("**** Load HMC samples ****")
    hmc_samples, kwargs = utils.load_hmc_samples(run_name)
    kwargs['device'] = device

    # Evenly space integer from N to M with K samples
    selected_idxs = np.linspace(500, 9900-1, N_ensemble, dtype=int)
    if N_ensemble <= 20:
        print(f"Out of {len(hmc_samples['params'])} samples, use idx: {selected_idxs}")
    else:
        print(f"Out of {len(hmc_samples['params'])} samples, use {N_ensemble} evenly spaced samples")
        
    selected_samples = [hmc_samples['params'][i] for i in selected_idxs]
    net = EnsNet(**kwargs)
    net.set_weights(selected_samples[-1])

    return net


def get_Ensemble_net_neg1_rebased_m_std(run_name, device, target_net, train_loader=None, without_rebasin=False):
    print("**** Load Ensemble ****")
    ens = utils.load_ensemble(run_name, device=device)
    print(f"Ensemble size: {ens.N_ensemble}")

    if without_rebasin:
        # Get the mean and std from selected_samples
        ws_m, ws_std = ens.get_weights_m_std()
        return ws_m, ws_std, ens.nets[-1]
    
    print("**** Rebase Ensemble ****")
    _, rb_ws_m, rb_ws_std, _ = ens.vri(train_loader, disable_rebasin=False)
    rrb_ws_m, rrb_ws_std = rebasin.weight_to_net_matching(rb_ws_m, 
                                                          target_net, 
                                                          train_loader=train_loader, 
                                                          params_perm_together=rb_ws_std)

    return rrb_ws_m, rrb_ws_std, ens.nets[-1]


def get_Ensemble_rebased_m_std(run_name, device, target_net, train_loader=None, without_rebasin=False):
    print("**** Load Ensemble ****")
    ens = utils.load_ensemble(run_name, device=device)
    print(f"Ensemble size: {ens.N_ensemble}")

    if without_rebasin:
        # Get the mean and std from selected_samples
        ws_m, ws_std = ens.get_weights_m_std()
        return ws_m, ws_std
    
    print("**** Rebase Ensemble ****")
    _, rb_ws_m, rb_ws_std, _ = ens.vri(train_loader, disable_rebasin=False)
    rrb_ws_m, rrb_ws_std = rebasin.weight_to_net_matching(rb_ws_m, 
                                                          target_net, 
                                                          train_loader=train_loader, 
                                                          params_perm_together=rb_ws_std)

    return rrb_ws_m, rrb_ws_std


def get_VI_rebased_m_std(run_name, device, target_net, train_loader=None, without_rebasin=False):
    print("**** Load VI ****")
    vi_net = utils.load_vi(run_name, device=device)

    if without_rebasin:
        # Get the mean and std from selected_samples
        vi_means, vi_stds = vi_net.get_means_stds()
        return vi_means, vi_stds

    print("**** Rebase VI ****")
    rb_vi_net = rebasin.vi_net_matching(vi_net, target_net, train_loader)
    rb_vi_means, rb_vi_stds = rb_vi_net.get_means_stds()

    return rb_vi_means, rb_vi_stds


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

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1000, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False)

    num_samples=100
    without_rebasin = False

    
    print(f"**** Get HMC rebased mean and std ****")
    # Get HMC rebased mean and std
    net0_hmc, m_hmc, std_hmc = get_HMC_rebased_m_std(run_name_hmc, device, train_loader=train_loader, N_ensemble=1000, without_rebasin=without_rebasin, rb2mean=True)
    # Save the net0_hmc, m_hmc, std_hmc in dir "./saved_models/"+run_name_hmc
    torch.save(net0_hmc.state_dict(), "./saved_models/"+run_name_hmc+"/rb2mean_net0_hmc.pt")
    np.savez("./saved_models/"+run_name_hmc+"/rb2mean_m_hmc_std_hmc.npz", m_hmc=m_hmc, std_hmc=std_hmc)
    print(f"Saved the net0_hmc, m_hmc, std_hmc in dir './saved_models/{run_name_hmc}'")


    # Get Ensemble rebased mean and std
    m_ens, std_ens= get_Ensemble_rebased_m_std(run_name_ens, device, target_net=net0_hmc, train_loader=train_loader, without_rebasin=without_rebasin)

    # Get VI rebased mean and std
    m_vi, std_vi = get_VI_rebased_m_std(run_name_vi, device, target_net=net0_hmc, train_loader=train_loader, without_rebasin=without_rebasin)

    # Save the rebased mean and std as numpy array to "./plot_data/pairplot_data.npz"
    results = {
        "m_hmc": to_numpy(m_hmc),
        "std_hmc": to_numpy(std_hmc),
        "m_ens": to_numpy(m_ens),
        "std_ens": to_numpy(std_ens),
        "m_vi": to_numpy(m_vi),
        "std_vi": to_numpy(std_vi),
    }
    if without_rebasin:
        save_name = "pairplot_data_MAP_without_rebasin.npz"
    elif not without_rebasin:
        save_name = "pairplot_data_MAP_rebase_to_HMC_mean.npz"

    np.savez("./plot_data/"+save_name, **results)
    print(f"Saved the rebased mean and std to './plot_data/{save_name}'")



