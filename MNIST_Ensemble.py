""" VI for MNIST
    Ref: https://github.com/probabilisticai/probai-2022 (Yingzhen)
"""

import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms


import utils
from MNIST_models import MNIST_Ensemble, evaluate

EPS = 1e-5  # define a small constant for numerical stability control


# plot the training curve
def plot_training_loss(logs, title):
    N_ensemble = logs.shape[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    for i in range(N_ensemble):
        ax1.plot(np.arange(logs.shape[1]), logs[i, :, 0], label='nll_{}'.format(i))
        ax2.plot(np.arange(logs.shape[1]), logs[i, :, 1], label='acc_{}'.format(i))
    for ax in [ax1, ax2]:
        ax.legend()
        ax.set_xlabel('epoch')
        ax1.set_title(title)
    plt.show()

if __name__ == '__main__':
    # setting up the MNIST dataset
    transform=transforms.Compose([
            transforms.ToTensor(),])
    train_data = datasets.MNIST('./data', train=True, download=False,
                        transform=transform)
    test_data = datasets.MNIST('./data', train=False, download=False,
                        transform=transform)
    # train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    kwargs = {
            'input_dim':784, 
            'output_dim':10, 
            'N_ensemble':5,
            'init_std': 0.05,
            'device': device}

    ens = MNIST_Ensemble(**kwargs)

    # start training
    learning_rate = 1e-3 # default: 1e-3, SGLD: 1e-4
    N_epochs = 50
    sigma = 1. # For MAP 1, for MLE 0

    
    run_name=f"test-Ensemble-N_ensemble_{kwargs['N_ensemble']}-Ep_{N_epochs}-Adam-MAP_sigma_{sigma}"

    # the training loop starts
    logs = ens.train(learning_rate, train_data, N_epochs, sigma=sigma)

    accuracy = evaluate(ens, test_loader, device)
    print('Test Accuracy: {}%'.format(accuracy))

    # save the ensemble
    print("**** Save ensemble ****")
    utils.save_ensemble(ens, kwargs, run_name)

    print("**** Load ensemble ****")
    new_ens = utils.load_ensemble(run_name)

    print("**** Test ensemble ****")
    accuracy = evaluate(new_ens, test_loader, device)
    print('Reload Test Accuracy: {}%'.format(accuracy))

    print("**** VRI ensemble ****")
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
    rb_net, rb_ws_m, rb_ws_std, rb_ws_B = new_ens.vri(train_loader, disable_rebasin=False)
    accuracy = evaluate(rb_net, test_loader, device)
    print('VRI Test Accuracy: {}%'.format(accuracy))