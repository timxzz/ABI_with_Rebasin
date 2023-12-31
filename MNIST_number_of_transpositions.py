# A pytorch MNIST classifier with 3 layers of neural network.
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim

import numpy as np

from tqdm import tqdm
import copy

import utils_runs
import rebasin
from MNIST_models import Net



class NetPermuter:
    def __init__(self, device, min_transposition, seed=42, batch_size=64):
        self.closest_permutation = rebasin.closest_permutation
        self.closest_permutation_by_activation = rebasin.closest_permutation_by_activation
        self.device = device
        self.min_transposition = min_transposition
        self.m0 = Net(device).to(device)
        self.m0_init = copy.deepcopy(self.m0)
        self.p1 = rebasin.generate_permutation_for_i_min_transposition(min_transposition, length=512)
        self.m1_init = self.permute_weights(self.m0, self.p1)
        self.m1 = copy.deepcopy(self.m1_init)

        # import mnist data for barrier evaluation
        torch.manual_seed(seed)
        transform = transforms.Compose([transforms.ToTensor()])
        train_data = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
        test_data = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    def permute_weights(self, model_in, perm):
        model = copy.deepcopy(model_in)
        with torch.no_grad():
            # Sample a random permutation matrix for the first layer
            w1 = model.fc1.weight # 512 x 784
            b1 = model.fc1.bias
            model.fc1.weight = nn.Parameter(w1[perm, :])
            model.fc1.bias = nn.Parameter(b1[perm])

            # Sample a random permutation matrix for the second layer
            w2 = model.fc2.weight # 10 x 512
            model.fc2.weight = nn.Parameter(w2[:, perm])
        return model


    def evaluate_model(self, model, use_train_data=False):
        if use_train_data:
            data_loader = self.train_loader
        else:
            data_loader = self.test_loader

        model.to(self.device) 
        model.eval()
        loss = 0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for images, labels in tqdm(data_loader, desc="Eval One", leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss += labels.size(0) * criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total, loss / total


    def interpolate_state_dicts(self, state_dict_1, state_dict_2, lamb):
        return {key: (1 - lamb) * state_dict_1[key] + lamb * state_dict_2[key]
                for key in state_dict_1.keys()}


    def get_barrier_mid(self, m0, m1):
        m0_acc_train, m0_loss_train = self.evaluate_model(m0, use_train_data=True)
        m0_acc_test, m0_loss_test = self.evaluate_model(m0, use_train_data=False)
        m1_acc_train, m1_loss_train = self.evaluate_model(m1, use_train_data=True)
        m1_acc_test, m1_loss_test = self.evaluate_model(m1, use_train_data=False)
        base_er_train = ((1.-m0_acc_train) + (1.-m1_acc_train)) / 2
        base_er_test = ((1.-m0_acc_test) + (1.-m1_acc_test)) / 2
        base_loss_train = (m0_loss_train + m1_loss_train) / 2
        base_loss_test = (m0_loss_test + m1_loss_test) / 2

        mid_lamb = 0.5
        sd1 = m0.state_dict().copy()
        sd2 = m1.state_dict().copy()
        mid_model = Net(self.device)
        mid_model.load_state_dict(self.interpolate_state_dicts(sd1, sd2, mid_lamb))
        acc_train, loss_train = self.evaluate_model(mid_model, use_train_data=True)
        acc_test, loss_test = self.evaluate_model(mid_model, use_train_data=False)

        er_train = 1. - acc_train
        er_test = 1. - acc_test

        return er_train-base_er_train, er_test-base_er_test, loss_train-base_loss_train, loss_test-base_loss_test

    def compare_models_weights(self, m0, m1, by_activation=False):
        '''
        1. Measure the number of permutations between the two models
        2. Compare the norm of the weights difference
        '''
        if by_activation:
            perm = self.closest_permutation_by_activation(m0, m1, self.train_loader)
        else:
            perm = self.closest_permutation(m0.fc1.weight, m1.fc1.weight)
        n_perm = rebasin.count_perm_dis(perm)

        with torch.no_grad():
            m0w1 = m0.fc1.weight
            m0b1 = m0.fc1.bias
            m0w2 = m0.fc2.weight
            m0b2 = m0.fc2.bias
            m1w1 = m1.fc1.weight
            m1b1 = m1.fc1.bias
            m1w2 = m1.fc2.weight
            m1b2 = m1.fc2.bias
            norm_w1 = torch.norm(m0w1-m1w1)
            norm_b1 = torch.norm(m0b1-m1b1)
            norm_w2 = torch.norm(m0w2-m1w2)
            norm_b2 = torch.norm(m0b2-m1b2)
            # flatten and concatenate the weights for m0 and m1
            flat_m0 = torch.cat((m0w1.flatten(), m0b1.flatten(), m0w2.flatten(), m0b2.flatten()))
            flat_m1 = torch.cat((m1w1.flatten(), m1b1.flatten(), m1w2.flatten(), m1b2.flatten()))
            norm_flat = torch.norm(flat_m0-flat_m1)

        return n_perm, norm_w1.item(), norm_b1.item(), norm_w2.item(), norm_b2.item(), norm_flat.item()







def train(model, model_name, seed, epochs=10, lr=0.001, batch_size=64, opti="SGD", train_list=False):

    torch.manual_seed(seed)
    # import mnist data
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=False, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    device = model.device

    # train the model
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    if opti == "SGD":
        # use sgd optimizer
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opti == "Adam":
        # use adam optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr)


    step_idx = 0
    for epoch in tqdm(range(epochs), desc=f"Train {model_name}", leave=False):
        running_loss = 0.0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += labels.size(0) * loss.item()
            total += labels.size(0)
            step_idx += 1

            
        # # save model checkpoint at the end of each epoch
        # torch.save(model.state_dict(), model_name + '_epoch_{}.pth'.format(epoch))


    train_loss = running_loss/total


    # test the model
    model.eval()

    correct = 0
    total = 0
    loss = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Test One", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            loss += labels.size(0) * criterion(outputs, labels).item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    tqdm.write('Accuracy of the network on the 10000 test images: {}'.format(100 * correct / total))

    test_loss = loss / total

    return model, train_loss, test_loss


def train_model_N_perm(num_trains, seed, device, epochs=10, lr=0.001, batch_size=64, opti="SGD", rebasin_by_activation=False):
    # Uniform samples
    # min_transpositions = np.random.randint(1, 512, num_trains)

    # Even spacing
    min_transpositions = np.linspace(1, 511, num_trains).astype(int)
    tqdm.write(f"Transpositions: {min_transpositions}")

    np.random.seed(seed)
    seeds = np.random.randint(0, 100000, num_trains*2)

    i = 0
    init_bars_er_train = []
    init_bars_er_test = []
    init_bars_loss_train = []
    init_bars_loss_test = []
    trained_bars_er_train = []
    trained_bars_er_test = []
    trained_bars_loss_train = []
    trained_bars_loss_test = []
    permed_bars_er_train = []
    permed_bars_er_test = []
    permed_bars_loss_train = []
    permed_bars_loss_test = []
    init_n_perms = []
    init_norm_w1s = []
    init_norm_b1s = []
    init_norm_w2s = []
    init_norm_b2s = []
    init_norm_flat = []
    trained_n_perms = []
    trained_norm_w1s = []
    trained_norm_b1s = []
    trained_norm_w2s = []
    trained_norm_b2s = []
    trained_norm_flat = []
    permed_n_perms = []
    permed_norm_w1s = []
    permed_norm_b1s = []
    permed_norm_w2s = []
    permed_norm_b2s = []
    permed_norm_flat = []
    m0_train_losses = []
    m1_train_losses = []
    m0_test_losses = []
    m1_test_losses = []


    for min_transposition in tqdm(min_transpositions, desc="Train List", leave=False):
        tqdm.write("***********************")
        tqdm.write(f"Initialise two models with one permuted using {min_transposition} transpositions")
        netpermuter = NetPermuter(device, min_transposition)
        m0 = netpermuter.m0
        m1 = netpermuter.m1
        init_bar_er_train, init_bar_er_test, init_bar_loss_train, init_bar_loss_test = netpermuter.get_barrier_mid(m0, m1)
        n_perm, norm_w1, norms_b1, norms_w2, norms_b2, norms_flat = netpermuter.compare_models_weights(m0, m1, by_activation=rebasin_by_activation)
        init_bars_er_train.append(init_bar_er_train)
        init_bars_er_test.append(init_bar_er_test)
        init_bars_loss_train.append(init_bar_loss_train)
        init_bars_loss_test.append(init_bar_loss_test)
        init_n_perms.append(n_perm)
        init_norm_w1s.append(norm_w1)
        init_norm_b1s.append(norms_b1)
        init_norm_w2s.append(norms_w2)
        init_norm_b2s.append(norms_b2)
        init_norm_flat.append(norms_flat)


        tqdm.write("Train two models")
        m0, m0_train_loss, m0_test_loss = train(m0, f"seed_{seed}-m0-{min_transposition}", seeds[i], epochs=epochs, lr=lr, batch_size=batch_size, opti=opti, train_list=True)
        m1, m1_train_loss, m1_test_loss = train(m1, f"seed_{seed}-m1-{min_transposition}", seeds[num_trains+i], epochs=epochs, lr=lr, batch_size=batch_size, opti=opti, train_list=True)
        trained_bar_er_train, trained_bar_er_test, trained_bar_loss_train, trained_bar_loss_test = netpermuter.get_barrier_mid(m0, m1)
        n_perm, norm_w1, norms_b1, norms_w2, norms_b2, norms_flat = netpermuter.compare_models_weights(m0, m1, by_activation=rebasin_by_activation)
        trained_bars_er_train.append(trained_bar_er_train)
        trained_bars_er_test.append(trained_bar_er_test)
        trained_bars_loss_train.append(trained_bar_loss_train)
        trained_bars_loss_test.append(trained_bar_loss_test)
        trained_n_perms.append(n_perm)
        trained_norm_w1s.append(norm_w1)
        trained_norm_b1s.append(norms_b1)
        trained_norm_w2s.append(norms_w2)
        trained_norm_b2s.append(norms_b2)
        trained_norm_flat.append(norms_flat)
        m0_train_losses.append(m0_train_loss)
        m1_train_losses.append(m1_train_loss)
        m0_test_losses.append(m0_test_loss)
        m1_test_losses.append(m1_test_loss)
        

        tqdm.write("Permute the trained models and compare weights")
        m0_p1 = netpermuter.permute_weights(m0, netpermuter.p1)
        permed_bar_er_train, permed_bar_er_test, permed_bar_loss_train, permed_bar_loss_test = netpermuter.get_barrier_mid(m0_p1, m1)
        n_perm, norm_w1, norms_b1, norms_w2, norms_b2, norms_flat = netpermuter.compare_models_weights(m0_p1, m1, by_activation=rebasin_by_activation)
        permed_bars_er_train.append(permed_bar_er_train)
        permed_bars_er_test.append(permed_bar_er_test)
        permed_bars_loss_train.append(permed_bar_loss_train)
        permed_bars_loss_test.append(permed_bar_loss_test)
        permed_n_perms.append(n_perm)
        permed_norm_w1s.append(norm_w1)
        permed_norm_b1s.append(norms_b1)
        permed_norm_w2s.append(norms_w2)
        permed_norm_b2s.append(norms_b2)
        permed_norm_flat.append(norms_flat)

        i += 1

    results = {
        "init_bars_er_train": init_bars_er_train,
        "init_bars_er_test": init_bars_er_test,
        "init_bars_loss_train": init_bars_loss_train,
        "init_bars_loss_test": init_bars_loss_test,
        "trained_bars_er_train": trained_bars_er_train,
        "trained_bars_er_test": trained_bars_er_test,
        "trained_bars_loss_train": trained_bars_loss_train,
        "trained_bars_loss_test": trained_bars_loss_test,
        "permed_bars_er_train": permed_bars_er_train,
        "permed_bars_er_test": permed_bars_er_test,
        "permed_bars_loss_train": permed_bars_loss_train,
        "permed_bars_loss_test": permed_bars_loss_test,
        "min_transpositions": min_transpositions,
        "init_n_perms": init_n_perms,
        "trained_n_perms": trained_n_perms,
        "permed_n_perms": permed_n_perms,
        "init_norm_w1s": init_norm_w1s,
        "init_norm_b1s": init_norm_b1s,
        "init_norm_w2s": init_norm_w2s,
        "init_norm_b2s": init_norm_b2s,
        "init_norm_flat": init_norm_flat,
        "trained_norm_w1s": trained_norm_w1s,
        "trained_norm_b1s": trained_norm_b1s,
        "trained_norm_w2s": trained_norm_w2s,
        "trained_norm_b2s": trained_norm_b2s,
        "trained_norm_flat": trained_norm_flat,
        "permed_norm_w1s": permed_norm_w1s,
        "permed_norm_b1s": permed_norm_b1s,
        "permed_norm_w2s": permed_norm_w2s,
        "permed_norm_b2s": permed_norm_b2s,
        "permed_norm_flat": permed_norm_flat,
        "m0_train_losses": m0_train_losses,
        "m1_train_losses": m1_train_losses,
        "m0_test_losses": m0_test_losses,
        "m1_test_losses": m1_test_losses,
    }


    return results





if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(device)

    # convert command line arguments to dictionary
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')

    parser.add_argument(
        "--dataset", choices=["BinaryMNIST", "MNIST", "CIFAR10", "ImageNet", "SVHN"],
        help="Choose the dataset for the experment.")
    parser.add_argument(
        "--optimizer", choices=["SGD", "Adam"],
        help="Choose the optimizer for the experment.")

    parser.add_argument(
        "--num_epochs", type=int, default=100,
        help="Number of epochs for training.")
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size for training.")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3,
        help="Learning rate.")

    parser.add_argument(
        "--num_perm_exps", type=int, default=1,
        help="Number of pairwise permutation experiments.")

    parser.add_argument(
        "--run_name", default="",
        help="Specified the name of the run.")
    parser.add_argument(
        "--run_batch_name", default="singles",
        help="Specified the name of the batch for runs if doing a batch grid search etc.")
    
    # boolean arguments for rebasin by activation, default is by weight and false
    parser.add_argument('--by_activation', dest='rebasin_by_activation', action='store_true')

    args = parser.parse_args()


    config_string = (
        f"-Epochs_{args.num_epochs}"
        f"-BatchSize_{args.batch_size}"
        f"-{args.dataset}"
        f"-{args.optimizer}"
        f"-Seed_{args.seed}"
        f"-NumPermExps_{args.num_perm_exps}"
        f"-ByActivation_{args.rebasin_by_activation}"
    )

    if args.run_name == "":
        run_name = config_string
    else:
        run_name = args.run_name + config_string
    print(run_name)

    out_dir = './runs/' + args.run_batch_name
    # Save training config
    utils_runs.save_train_config(out_dir, run_name, vars(args))


    # Set random seeds
    torch.manual_seed(args.seed)

    bars_nomrs_results = train_model_N_perm(args.num_perm_exps, args.seed, device, 
                                            epochs=args.num_epochs, 
                                            lr=args.learning_rate, 
                                            batch_size=args.batch_size, 
                                            opti=args.optimizer,
                                            rebasin_by_activation=args.rebasin_by_activation)

    # Save results
    utils_runs.save_train_results(out_dir, bars_nomrs_results, run_name, "barriers_norms_vs_num_of_permutations")
