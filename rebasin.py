import torch
import torch.nn as nn
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
import numpy as np
import copy

def cost_matrix(w1_tensor, w2_tensor):
    """
    @param w1_tensor: a tensor of shape (n, m)
    @param w2_tensor: a tensor of shape (n, m)
    """
    w1 = w1_tensor.cpu().clone().detach().numpy()
    w2 = w2_tensor.cpu().clone().detach().numpy()
    n = len(w1)
    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # 1. base on the sum of squared differences between the rows
            # cost[i, j] = np.sum((w1[i] - w2[j])**2)

            # 2. base on the norm of the difference between the rows
            # cost[i, j] = np.linalg.norm(w1[i] - w2[j])

            # 3. base on the cosine similarity between the rows
            cost[i, j] = 1 - np.dot(w1[i], w2[j]) / (np.linalg.norm(w1[i]) * np.linalg.norm(w2[j]))

    return cost



def acts_cost_matrix(m1, m2, train_loader):
    device = m1.device
    cost = torch.zeros((m1.h_dim, m2.h_dim), device=device)
    with torch.no_grad():
        for images, _ in tqdm(train_loader, desc="Get activations", leave=False):
            images= images.to(device)
            acts1 = m1.get_activations(images)
            acts2 = m2.get_activations(images)
            cost = cost + acts1.T @ acts2
    return -cost.cpu().numpy()

def closest_permutation_by_activation(m1, m2, train_loader):
    """
    Find the permutation matrix P that minimizes the cost between two activations matrices
    such that: ||A1 - P * A2||_F^2 is minimized
    """
    cost = acts_cost_matrix(m1, m2, train_loader)
    row_ind, col_ind = linear_sum_assignment(cost)

    return col_ind

def closest_permutation(w1, w2):
    """
    Find the permutation matrix P that minimizes the cost between two weight matrices
    such that: w1 = P * w2
    """
    cost = cost_matrix(w1, w2)
    row_ind, col_ind = linear_sum_assignment(cost)
    return col_ind

def permute_weights(model_in, perm):
    model = copy.deepcopy(model_in)
    with torch.no_grad():
        # Sample a random permutation matrix for the first layer
        w1 = model.fc1.weight # 200 x 1
        b1 = model.fc1.bias
        model.fc1.weight = nn.Parameter(w1[perm, :])
        model.fc1.bias = nn.Parameter(b1[perm])

        # Sample a random permutation matrix for the second layer
        w2 = model.fc2.weight # 1 x 200
        model.fc2.weight = nn.Parameter(w2[:, perm])
    return model

def permute_vi_weights(model_in, perm):
    model = copy.deepcopy(model_in)
    with torch.no_grad():
        # Sample a random permutation matrix for the first layer
        w1 = model.fc1.weight_mean # 200 x 1
        b1 = model.fc1.bias_mean
        w1_std = model.fc1._weight_std_param
        b1_std = model.fc1._bias_std_param
        model.fc1.weight_mean = nn.Parameter(w1[perm, :])
        model.fc1.bias_mean = nn.Parameter(b1[perm])
        model.fc1._weight_std_param = nn.Parameter(w1_std[perm, :])
        model.fc1._bias_std_param = nn.Parameter(b1_std[perm])


        # Sample a random permutation matrix for the second layer
        w2 = model.fc2.weight_mean # 1 x 200
        w2_std = model.fc2._weight_std_param
        model.fc2.weight_mean = nn.Parameter(w2[:, perm])
        model.fc2._weight_std_param = nn.Parameter(w2_std[:, perm])
    return model

def count_perm_dis(P):
    flags = torch.zeros(P.shape[0])
    disjoints = []
    i = 0
    while torch.sum(flags) < P.shape[0]:
        cycle = []
        j = i
        while flags[j] == 0:
            cycle.append(j)
            flags[j] = 1
            j = P[j]
        i += 1

        if len(cycle) > 0:
            disjoints.append(cycle)
        
    counts = sum([(len(cycle)-1) for cycle in disjoints])
    return counts

def nets_weight_matching(nets, target_net=None, train_loader=None):
    if train_loader is None:
        tqdm.write("Weight Matching")
    else:
        tqdm.write("Activation Matching")
    rebased_nets = []
    NoTs_list = []
    # nets = nets[::-1]
    for net in tqdm(nets, desc="Matching Nets", leave=False):
        if target_net is None:
            target_net = net
            rebased_nets.append(net)
        else:
            if train_loader is None:
                perm = closest_permutation(target_net.fc1.weight, net.fc1.weight)
            else:
                perm = closest_permutation_by_activation(target_net, net, train_loader)
            n_trans = count_perm_dis(perm)
            NoTs_list.append(n_trans)
            rebased_nets.append(permute_weights(net, perm))
    tqdm.write(f"Number of Transpositions: {NoTs_list}")
    return rebased_nets

def weight_to_net_matching(weight, target_net, train_loader=None, params_perm_together=None):
    if train_loader is None:
        tqdm.write("Weight Matching")
    else:
        tqdm.write("Activation Matching")

    net = copy.deepcopy(target_net)
    net.set_weights(weight)

    if train_loader is None:
        perm = closest_permutation(target_net.fc1.weight, net.fc1.weight)
    else:
        perm = closest_permutation_by_activation(target_net, net, train_loader)
    n_trans = count_perm_dis(perm)
    tqdm.write(f"Number of Transposition: {n_trans}")
    rebased_weight = permute_weights(net, perm).get_flat_weights()
    if params_perm_together is None:
        return rebased_weight
    else:
        net.set_weights(params_perm_together)
        rebased_params = permute_weights(net, perm).get_flat_weights()
        return rebased_weight, rebased_params

def vi_net_matching(vi_net, target_net, train_loader):
    print("Matching VI net")
    perm = closest_permutation_by_activation(target_net, vi_net, train_loader)
    n_trans = count_perm_dis(perm)
    print(f"Number of Transposition: {n_trans}")
    rebased_vi_net = permute_vi_weights(vi_net, perm)
    return rebased_vi_net

def vi_by_ensemble(nets):
    # Get the weights of each network
    weights = torch.stack([net.get_flat_weights().cpu().clone().detach() for net in nets], dim=0)

    # Calculate the mean of the weights
    weights_mean = weights.mean(dim=0)
    weights_std = weights.std(dim=0)

    # Calculate the empirical covariance matrix
    weights_centered = weights - weights_mean # shape: (k, d), k << d, k is the number of nets, d is the number of weights
    # weights_cov = weights_centered.T @ weights_centered / (len(nets) - 1)
    
    # Use SVD to calculate the B of C=BB^T
    U, S, Vt = torch.linalg.svd(weights_centered.T, full_matrices=False)
    weights_B = U @ torch.diag(S) / np.sqrt(len(nets) - 1)
    # weights_cov = weights_B @ weights_B.T

    return weights_mean, weights_std, weights_B


def generate_permutation_for_i_min_transposition(i, length):
    assert i < length and i > 0
    perm = torch.arange(length)
    candidates = torch.arange(length)
    n_cycles = length - i
    cycle_groups = np.ones(n_cycles, dtype=int)
    n_assignment_left = length - n_cycles

    # Randomly assign the size of each cycle group
    while n_assignment_left > 0:
        cycle_groups[np.random.randint(0, n_cycles)] += 1
        n_assignment_left -= 1

    # Randomly choose the elements and put them into the cycle groups
    for cycle_group in cycle_groups:
        selected_idxs = torch.multinomial(torch.ones_like(candidates, dtype=torch.float), cycle_group, replacement=False)
        cycle = candidates[selected_idxs]
        candidates = torch.unique(torch.masked_select(candidates, ~torch.isin(candidates, cycle))) #torch.setdiff1d(candidates, cycle)
        perm[cycle] = torch.roll(cycle, 1, dims=0)

    assert count_perm_dis(perm) == i

    return perm