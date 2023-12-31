import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import copy
from tqdm import tqdm

import rebasin


def to_numpy(x):
    # if x is already a numpy array, return x
    if isinstance(x, np.ndarray):
        return x
    return x.detach().cpu().numpy() # convert a torch tensor to a numpy array

# test the Ensemble
def evaluate(model, dataloader, device):
    accuracy = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device); y = y.to(device)
            # check if model is an MNIST_Ensemble object
            if isinstance(model, MNIST_Ensemble):
                # the "predict" function: by default dropout is on and we use K > 1 MC samples
                y_pred = model.predict_ens(x, reduce_mean=True)
            else:
                y_pred = model.predict(x)
            pred = y_pred.data.max(1, keepdim=True)[1] # get the index of the max probability
            accuracy += pred.eq(y.data.view_as(pred)).float().cpu().sum()
        accuracy = accuracy / len(dataloader.dataset) * 100 # accuracy in percentage
    return to_numpy(accuracy)



class Linear(nn.Module):
    """Applies a linear transformation to the incoming data: y = xW^T + b.
    """

    def __init__(self, dim_in, dim_out, init_std=0.05,
                device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.dim_in = dim_in  # dimension of network layer input
        self.dim_out = dim_out  # dimension of network layer output

        # define and initialise the parameters
        self.weight = nn.Parameter(torch.empty((dim_out, dim_in), **factory_kwargs))
        self.bias = nn.Parameter(torch.empty(dim_out, **factory_kwargs))
        self.reset_parameters(init_std)

    def reset_parameters(self, init_std=0.05):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = self.dim_in ** -0.5
        nn.init.uniform_(self.bias, -bound, bound)

    # forward pass with Monte Carlo (MC) sampling
    def forward(self, input):
        return F.linear(input, self.weight, self.bias)


class EnsNet(nn.Module):
    def __init__(self, input_dim, output_dim, **layer_kwargs):
        super().__init__()
        
        self.h_dim = 512
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = layer_kwargs['device']

        self.fc1 = Linear(self.input_dim, self.h_dim, **layer_kwargs)
        self.fc2 = Linear(self.h_dim, self.output_dim, **layer_kwargs)

    def loss_batch(self, x, y, sigma=0, training_set_size=None):
        y_logit = self.forward(x)
        nll = F.nll_loss(F.log_softmax(y_logit, dim=-1), y)

        # Compute regularization term for the weights
        reg_term = 0
        if sigma > 0:
            for param in self.parameters():
                reg_term += 0.5 * (param ** 2).sum() / (sigma ** 2)
            reg_term = reg_term / training_set_size

        # Combine the NLL loss and the regularization term
        total_loss = nll + reg_term

        # training accruacy (on a mini-batch)
        pred = y_logit.data.max(1, keepdim=True)[1] # get the index of the max logit
        acc = pred.eq(y.data.view_as(pred)).float().cpu().mean()
        logging = [to_numpy(total_loss), to_numpy(acc)]

        return total_loss, logging

    # define the prediction function
    def predict(self, x_test):
        y_pred = F.softmax(self.forward(x_test), dim=-1)
        return y_pred

    def forward(self, x):
        h = x.view(-1, 784)
        h = F.relu(self.fc1(h))
        h = self.fc2(h) # Logits
        return h

    def get_activations(self, x):
        x = x.view(-1, 784)
        h = F.relu(self.fc1(x))
        return h

    @staticmethod
    def load_flatten_weights_to_net(net, flattened_params):
        if flattened_params.dim() != 1:
            raise ValueError('Expecting a 1d flattened_params')
        i = 0
        for param in net.parameters():
            param.data = flattened_params[i:i+param.numel()].view(param.size())
            i += param.numel()

    def set_weights(self, flattened_params):
        self.load_flatten_weights_to_net(self, flattened_params)

    def get_weights_names(self):
        return [name for name, _ in self.named_parameters()]

    def get_flat_weights(self):
        return torch.cat([param.data.flatten() for param in self.parameters()])

    def get_weights_numels(self):
        return [param.data.numel() for param in self.parameters()]
    
    def get_weights_shapes(self):
        return [param.data.shape for param in self.parameters()]



class MNIST_Ensemble():
    def __init__(self, input_dim, output_dim, N_ensemble, **layer_kwargs):
        super().__init__()
        
        self.N_ensemble = N_ensemble
        self.device = layer_kwargs['device']
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_kwargs = layer_kwargs

        self.nets = []
        for i in range(N_ensemble):
            self.nets.append(EnsNet(input_dim, output_dim, **layer_kwargs))
        self.net_flat_weights_shape = self.nets[0].get_flat_weights().shape
        self.net_weights_numels = self.nets[0].get_weights_numels()
        self.net_weights_shapes = self.nets[0].get_weights_shapes()

    def set_weights(self, flattened_params_list):
        for i in range(self.N_ensemble):
            self.nets[i].set_weights(flattened_params_list[i])

    def get_weights(self):
        return torch.stack([net.get_flat_weights() for net in self.nets], dim=0)
    
    def get_weights_m_std(self):
        ws = self.get_weights()
        ws_m = ws.mean(dim=0)
        ws_std = ws.std(dim=0)
        return ws_m, ws_std

    # define the prediction function using N_ensemble samples
    def predict_ens(self, x_test, reduce_mean=True, nets=None):
        y_pred = []
        if nets is None:
            nets = self.nets
        N_nets = len(nets)
        for i in range(N_nets):
            net = self.nets[i]
            net.eval()
            net.to(self.device)
            y_pred.append(net.predict(x_test))
        # shape (K, batch_size, y_dim) or (batch_size, y_dim) if K = 1
        y_pred = torch.stack(y_pred, dim=0).squeeze(0)
        if reduce_mean and N_nets > 1:
            y_pred = y_pred.mean(0)
        return y_pred

    def train_step(self, net, opt, dataloader, sigma=0):
        logs = []
        for _, (x, y) in enumerate(dataloader):
            x = x.to(self.device); y = y.to(self.device)
            opt.zero_grad() # opt is the optimiser
            loss, logging = net.loss_batch(x, y, sigma=sigma, training_set_size=len(dataloader.dataset))
            loss.backward()
            opt.step()
            logs.append(logging)
        return np.array(logs)

    def train(self, learning_rate, train_data, N_epochs=2000, verbose=True, opt_name='Adam', sigma=0):
        assert opt_name in ['Adam', 'SGD', 'SGD_momentum']
        logs = []
        for K_id in range(self.N_ensemble):
            net = self.nets[K_id]
            net.train()
            net.to(self.device)
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
            if opt_name == 'Adam':
                opt = torch.optim.Adam(net.parameters(), lr=learning_rate)
            elif opt_name == 'SGD':
                opt = torch.optim.SGD(net.parameters(), lr=learning_rate)
            elif opt_name == 'SGD_momentum':
                opt = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
            logs_K = []
            for i in range(N_epochs):
                logs_epoch = self.train_step(net, opt, train_loader, sigma=sigma)
                if verbose:
                    tqdm.write("K_id {}, Epoch {}, last mini-batch nll={}, acc={}".format(
                        K_id, i+1, logs_epoch[-1][0], logs_epoch[-1][1]))
                logs_K.append(logs_epoch)
            logs.append(np.concatenate(logs_K, axis=0))
        return np.stack(logs, axis=0)

    def assign_weights_to_net(self, net, weights):
        i = 0
        for param in net.parameters():
            param.data = weights[i:i+param.numel()].view(param.size())
            i += param.numel()

    def rebasin(self, target_net=None, train_loader=None):
        return rebasin.nets_weight_matching(self.nets, target_net=target_net, train_loader=train_loader)

    def vri(self, train_loader, target_net=None, disable_rebasin=False):

        if disable_rebasin:
            rebased_nets = self.nets
        else:
            rebased_nets = self.rebasin(target_net=target_net, train_loader=train_loader)
        rb_ws_m, rb_ws_std, rb_ws_B = rebasin.vi_by_ensemble(rebased_nets) 

        # deep copy one net from self.nets[0]
        net = copy.deepcopy(self.nets[0])
        # assign the weights to the net
        self.assign_weights_to_net(net, rb_ws_m)

        return net, rb_ws_m, rb_ws_std, rb_ws_B

    def reparameterize(self, mu, B, num_samples=10):
        # mu: (d,)
        # B: (d, k)
        # return: (num_samples, d)
        k = B.shape[1]
        z = torch.randn(num_samples, k).to(self.device)
        # duplicate mu to match the shape (d, num_samples)
        mu = mu.unsqueeze(1).repeat(1, num_samples)
        ws = mu + B @ z.T
        return ws.T

    def sample_from_vri(self, rb_ws_m, rb_ws_B, num_samples=10):
        # rb_ws_m: (d,)
        # rb_ws_B: (d, k)

        ws = self.reparameterize(rb_ws_m, rb_ws_B, num_samples=num_samples)
        sample_nets = []
        for i in range(num_samples):
            # deep copy one net from self.nets[0]
            net = copy.deepcopy(self.nets[0])
            sample_weights = ws[i]
            # assign the weights to the net
            self.assign_weights_to_net(net, sample_weights)
            sample_nets.append(net)

        return sample_nets

    def sample_from_mf(self, rb_ws_m, rb_ws_std, num_samples=10):
        # rb_ws_m: (d,)
        # rb_ws_std: (d,)

        q_ws = D.Normal(rb_ws_m, rb_ws_std)
        sample_nets = []
        for i in range(num_samples):
            # deep copy one net from self.nets[0]
            net = copy.deepcopy(self.nets[0])
            sample_weights = q_ws.sample()
            # assign the weights to the net
            self.assign_weights_to_net(net, sample_weights)
            sample_nets.append(net)

        return sample_nets


# ------------------------------------------------------------------
# ------------------------------------------------------------------

EPS = 1e-5  # define a small constant for numerical stability control

class MFVILinear(nn.Module):
    """Applies a linear transformation to the incoming data: y = xW^T + b, where
    the weight W and bias b are sampled from the q distribution.
    """

    def __init__(self, dim_in, dim_out, prior_weight_std=1.0, prior_bias_std=1.0, init_std=0.05,
                 sqrt_width_scaling=False, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.dim_in = dim_in  # dimension of network layer input
        self.dim_out = dim_out  # dimension of network layer output

        # define the trainable variational parameters for q distribtuion
        # first define and initialise the mean parameters
        self.weight_mean = nn.Parameter(torch.empty((dim_out, dim_in), **factory_kwargs))
        self.bias_mean = nn.Parameter(torch.empty(dim_out, **factory_kwargs))
        self._weight_std_param = nn.Parameter(torch.empty((dim_out, dim_in), **factory_kwargs))
        self._bias_std_param = nn.Parameter(torch.empty(dim_out, **factory_kwargs))
        self.reset_parameters(init_std)

        # define the prior parameters (for prior p, assume the mean is 0)
        prior_mean = 0.0
        if sqrt_width_scaling:  # prior variance scales as 1/dim_in
            prior_weight_std /= self.dim_in ** 0.5
        # prior parameters are registered as constants
        self.register_buffer('prior_weight_mean', torch.full_like(self.weight_mean, prior_mean))
        self.register_buffer('prior_weight_std', torch.full_like(self._weight_std_param, prior_weight_std))
        self.register_buffer('prior_bias_mean', torch.full_like(self.bias_mean, prior_mean))
        self.register_buffer('prior_bias_std', torch.full_like(self._bias_std_param, prior_bias_std))

    def set_params(self, flat_params_mean, flat_params_std):
        # split the flattened mean and std for weight and bias
        # then assign the values to the mean and std parameters
        dim_in = self.dim_in
        dim_out = self.dim_out
        self.weight_mean.data = flat_params_mean[:dim_out*dim_in].view(dim_out, dim_in)
        self.bias_mean.data = flat_params_mean[dim_out*dim_in:]
        self._weight_std_param.data = flat_params_std[:dim_out*dim_in].view(dim_out, dim_in)
        self._bias_std_param.data = flat_params_std[dim_out*dim_in:]

    def extra_repr(self):
        s = "dim_in={}, dim_in={}, bias=True".format(self.dim_in, self.dim_out)
        weight_std = self.prior_weight_std.data.flatten()[0]
        if torch.allclose(weight_std, self.prior_weight_std):
            s += f", weight prior std={weight_std.item():.2f}"
        bias_std = self.prior_bias_std.flatten()[0]
        if torch.allclose(bias_std, self.prior_bias_std):
            s += f", bias prior std={bias_std.item():.2f}"
        return s

    def reset_parameters(self, init_std=0.05):
        nn.init.kaiming_uniform_(self.weight_mean, a=math.sqrt(5))
        bound = self.dim_in ** -0.5
        nn.init.uniform_(self.bias_mean, -bound, bound)
        _init_std_param = np.log(init_std)
        self._weight_std_param.data = torch.full_like(self.weight_mean, _init_std_param)
        self._bias_std_param.data = torch.full_like(self.bias_mean, _init_std_param)

    # define the q distribution standard deviations with property decorator
    @property
    def weight_std(self):
        return torch.clamp(torch.exp(self._weight_std_param), min=EPS)

    @property
    def bias_std(self):
        return torch.clamp(torch.exp(self._bias_std_param), min=EPS)

    # KL divergence KL[q||p] between two Gaussians
    def kl_divergence(self):
        q_weight = D.Normal(self.weight_mean, self.weight_std)
        p_weight = D.Normal(self.prior_weight_mean, self.prior_weight_std)
        kl = D.kl_divergence(q_weight, p_weight).sum()
        q_bias = D.Normal(self.bias_mean, self.bias_std)
        p_bias = D.Normal(self.prior_bias_mean, self.prior_bias_std)
        kl += D.kl_divergence(q_bias, p_bias).sum()
        return kl

    # forward pass with Monte Carlo (MC) sampling
    def forward(self, input, sample=True):
        if sample:
            weight = self._normal_sample(self.weight_mean, self.weight_std)
            bias = self._normal_sample(self.bias_mean, self.bias_std)
        else:
            weight = self.weight_mean
            bias = self.bias_mean
        return F.linear(input, weight, bias)

    def _normal_sample(self, mean, std):
        return mean + torch.randn_like(mean) * std

    def get_means_stds(self):
        w_m = self.weight_mean.cpu().clone().detach()
        w_s = self.weight_std.cpu().clone().detach()
        b_m = self.bias_mean.cpu().clone().detach()
        b_s = self.bias_std.cpu().clone().detach()
        means = torch.cat((w_m.flatten(), b_m.flatten()), dim=-1)
        stds = torch.cat((w_s.flatten(), b_s.flatten()), dim=-1)
        return means, stds


class MNIST_VI(nn.Module):
    def __init__(self, input_dim, output_dim, **layer_kwargs):
        super().__init__()
        
        self.h_dim = 512
        self.device = layer_kwargs['device']
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.fc1 = MFVILinear(input_dim, self.h_dim, **layer_kwargs)
        self.fc2 = MFVILinear(self.h_dim, output_dim, **layer_kwargs)

    # collect the kl divergence for all MFVILinear layers
    def kl_divergence(self):
        kl = 0.0
        kl += self.fc1.kl_divergence()
        kl += self.fc2.kl_divergence()
        return kl

    def nelbo_batch(self, x, y, N_data, beta=1.0):
        y_logit = self.forward(x)
        nll = F.nll_loss(F.log_softmax(y_logit, dim=-1), y)
        kl = self.kl_divergence()
        nelbo = N_data * nll + beta * kl

        # training accruacy (on a mini-batch)
        pred = y_logit.data.max(1, keepdim=True)[1] # get the index of the max logit
        acc = pred.eq(y.data.view_as(pred)).float().cpu().mean()
        logging = [to_numpy(nll), to_numpy(acc), to_numpy(kl)]

        return nelbo, logging

    # define the prediction function with Monte Carlo sampling using K samples
    def predict(self, x_test, K=1, reduce_mean=True, sample=True):
        y_pred = []
        for _ in range(K):
            y_pred.append(F.softmax(self.forward(x_test, sample=sample), dim=-1))
        # shape (K, batch_size, y_dim) or (batch_size, y_dim) if K = 1
        y_pred = torch.stack(y_pred, dim=0).squeeze(0)
        if reduce_mean and K > 1:
            y_pred = y_pred.mean(0)
        return y_pred

    def forward(self, x, sample=True):
        h = x.view(-1, 784)
        h = F.relu(self.fc1(h, sample=sample))
        h = self.fc2(h, sample=sample) # Logits
        return h

    def get_activations(self, x, K=10):
        acts = []
        for _ in range(K):
            x = x.view(-1, 784)
            h = F.relu(self.fc1(x))
            acts.append(h)
        acts = torch.stack(acts, dim=0).squeeze(0)
        if K > 1:
            acts = acts.mean(0)
        return acts

    def get_means_stds(self):
        fc1_m, fc1_s = self.fc1.get_means_stds()
        fc2_m, fc2_s = self.fc2.get_means_stds()
        means = torch.cat((fc1_m, fc2_m), dim=-1)
        stds = torch.cat((fc1_s, fc2_s), dim=-1)
        return means, stds
    
    def set_params(self, flat_params_mean, flat_params_std):
        dim_in = self.input_dim
        dim_h = self.h_dim
        fc1_dim = dim_h * dim_in + dim_h
        self.fc1.set_params(flat_params_mean[:fc1_dim], flat_params_std[:fc1_dim])
        self.fc2.set_params(flat_params_mean[fc1_dim:], flat_params_std[fc1_dim:])

    


# ------------------------------------------------------------------
# ------------------------------------------------------------------

# For evaluate number of transpositions

class Net(nn.Module):
    def __init__(self, device):
        super(Net, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 10)
        self.h_dim = 512

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # Logits
        return x

    def get_activations(self, x):
        x = x.view(-1, 784)
        h = F.relu(self.fc1(x))
        return h