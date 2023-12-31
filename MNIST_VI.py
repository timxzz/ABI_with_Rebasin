""" VI for MNIST
    Ref: https://github.com/probabilisticai/probai-2022 (Yingzhen)
"""

import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms


import utils
from MNIST_models import MNIST_VI, to_numpy


def train_bnn_step(net, opt, dataloader, device, beta):
    logs = []
    N_data = len(dataloader.dataset)
    for _, (x, y) in enumerate(dataloader):
        x = x.to(device); y = y.to(device)
        opt.zero_grad() # opt is the optimiser
        loss, logging = net.nelbo_batch(x, y, N_data, beta)
        loss.backward()
        opt.step()
        logs.append(logging)
    return np.array(logs)

def train_bnn(net, opt, dataloader, device, N_epochs=2000, beta=1.0, verbose=True):
    net.train()
    logs = []
    for i in range(N_epochs):
        logs_epoch = train_bnn_step(net, opt, dataloader, device, beta)
        if verbose:
            print("Epoch {}, last mini-batch nll={}, acc={}, kl={}".format(
                i+1, logs_epoch[-1][0], logs_epoch[-1][1], logs_epoch[-1][2]))
        logs.append(logs_epoch)
    return np.concatenate(logs, axis=0)

# test the BNN
def evaluate(model, dataloader, device, K=50, sample=True):
    accuracy = 0
    predictive_dist = []
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device); y = y.to(device)
            # the "predict" function: by default dropout is on and we use K > 1 MC samples
            y_pred = model.predict(x, K, reduce_mean=True, sample=sample)
            predictive_dist.append(y_pred)
            pred = y_pred.data.max(1, keepdim=True)[1] # get the index of the max probability
            accuracy += pred.eq(y.data.view_as(pred)).float().cpu().sum()
        accuracy = accuracy / len(dataloader.dataset) * 100 # accuracy in percentage
        predictive_dist = torch.cat(predictive_dist, dim=0)
    return to_numpy(accuracy), predictive_dist.cpu().numpy()

# plot the training curve
def plot_training_loss(logs, title):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    ax1.plot(np.arange(logs.shape[0]), logs[:, 0], 'r-', label='nll')
    ax2.plot(np.arange(logs.shape[0]), logs[:, 1], 'r-', label='acc')
    ax3.plot(np.arange(logs.shape[0]), logs[:, 2], 'r-', label='KL')
    for ax in [ax1, ax2, ax3]:
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
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    kwargs = {
                'input_dim':784,
                'output_dim':10,
                'prior_weight_std': 1.0,
                'prior_bias_std': 1.0,
                'sqrt_width_scaling': False,
                'init_std': 0.05,
                'device': device}

    net = MNIST_VI(**kwargs).to(device)

    # start training
    learning_rate = 1e-3
    N_epochs = 50
    run_name = "test-VI"
    beta = 1.0
    opt = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # the training loop starts
    logs = train_bnn(net, opt, train_loader, device, N_epochs, beta)

    plot_training_loss(logs, title='MFVI')

    accuracy, _ = evaluate(net, test_loader, device, K=50)
    print('Test Accuracy: {}%'.format(accuracy))

    # save the ensemble
    print("**** Save VI ****")
    utils.save_vi(net, kwargs, run_name)

    print("**** Load VI ****")
    new_net = utils.load_vi(run_name)

    print("**** Test VI ****")
    accuracy, _ = evaluate(new_net, test_loader, device)
    print('Reload Test Accuracy: {}%'.format(accuracy))