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


class NetMatching:
    def __init__(self, device, rebasin_by_activation=False, seed=42, batch_size=64):
        self.closest_permutation = rebasin.closest_permutation
        self.closest_permutation_by_activation = rebasin.closest_permutation_by_activation
        self.rebasin_by_activation = rebasin_by_activation
        self.device = device
        self.m0 = Net(device).to(device)
        self.m0_init = copy.deepcopy(self.m0)
        self.m1 = Net(device).to(device)
        self.m1_init = copy.deepcopy(self.m1)

        # import mnist data for barrier evaluation
        torch.manual_seed(seed)
        transform = transforms.Compose([transforms.ToTensor()])
        train_data = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
        test_data = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
        self.train_loader = DataLoader(train_data, batch_size=1000, shuffle=True)
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

    def compare_models(self, m0, m1, tag=None):
        norms = []
        with_wd_NoTs = True
        with torch.no_grad():
            if tag is not None:
                with_wd_NoTs = False
                m0w1 = m0.fc1.weight
                m0b1 = m0.fc1.bias
                m0w2 = m0.fc2.weight
                m0b2 = m0.fc2.bias
                m1w1 = m1.fc1.weight
                m1b1 = m1.fc1.bias
                m1w2 = m1.fc2.weight
                m1b2 = m1.fc2.bias
                norms.append(torch.norm(m0w1-m1w1))
                norms.append(torch.norm(m0b1-m1b1))
                norms.append(torch.norm(m0w2-m1w2))
                norms.append(torch.norm(m0b2-m1b2))


            lambs, lmc_train_loss, lmc_test_loss, lmc_train_er, lmc_test_er, _, \
                w1_diff_norms, w2_diff_norms, b1_diff_norms, b2_diff_norms, flat_diff_norms, NoTs_mlamb_m0  = self.get_barrier(m0, m1, with_wd_NoTs)


        return (lambs, lmc_train_loss, lmc_test_loss, lmc_train_er, lmc_test_er, 
                w1_diff_norms, w2_diff_norms, b1_diff_norms, b2_diff_norms, flat_diff_norms, NoTs_mlamb_m0)


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
            for images, labels in tqdm(data_loader, desc="Eval One Interpolation", leave=False):
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


    def get_barrier(self, m0, m1, with_wdiff_NoTs=False):
        lambs = np.linspace(0, 1, 11)
        # lambs = np.linspace(0, 1, 3)
        model_eval = Net(self.device)

        # Loss Barrier, weight diff,  & Number of Transpositions
        # Note: use m0 as the anchor point
        sd_list = []
        result_train_loss = []
        result_test_loss = []
        result_train_er = []
        result_test_er = []
        w1_diff_norms = []
        w2_diff_norms = []
        b1_diff_norms = []
        b2_diff_norms = []
        flat_diff_norms = []
        NoTs_mlamb_m0 = []
        for i in tqdm(range(len(lambs)), desc="Eval Barrier", leave=False):
            sd1 = m0.state_dict().copy()
            sd2 = m1.state_dict().copy()
            sd_lamb = self.interpolate_state_dicts(sd1, sd2, lambs[i])
            sd_list.append(sd_lamb)
            model_eval.load_state_dict(sd_lamb)
            acc_train, loss_train = self.evaluate_model(model_eval, use_train_data=True)
            result_train_loss.append(loss_train)
            result_train_er.append(1. - acc_train)
            acc_test, loss_test = self.evaluate_model(model_eval, use_train_data=False)
            result_test_loss.append(loss_test)
            result_test_er.append(1. - acc_test)

            if with_wdiff_NoTs:
                w1_diff_norms.append(torch.norm(sd1["fc1.weight"] - sd_lamb["fc1.weight"]).item())
                b1_diff_norms.append(torch.norm(sd1["fc1.bias"] - sd_lamb["fc1.bias"]).item())
                w2_diff_norms.append(torch.norm(sd1["fc2.weight"] - sd_lamb["fc2.weight"]).item())
                b2_diff_norms.append(torch.norm(sd1["fc2.bias"] - sd_lamb["fc2.bias"]).item())
                # flatten and concatenate the weights for sd1 and sd_lamb
                flat_sd1 = torch.cat([sd1["fc1.weight"].flatten(), sd1["fc1.bias"].flatten(),
                                    sd1["fc2.weight"].flatten(), sd1["fc2.bias"].flatten()])
                flat_sdlamb = torch.cat([sd_lamb["fc1.weight"].flatten(), sd_lamb["fc1.bias"].flatten(),
                                    sd_lamb["fc2.weight"].flatten(), sd_lamb["fc2.bias"].flatten()])
                flat_diff_norms.append(torch.norm(flat_sd1 - flat_sdlamb).item())

                if self.rebasin_by_activation:
                    perm = self.closest_permutation_by_activation(m0, model_eval, self.train_loader)
                else:
                    perm = self.closest_permutation(m0.fc1.weight, model_eval.fc1.weight)
                NoTs_mlamb_m0.append(rebasin.count_perm_dis(perm))



        return (lambs, result_train_loss, result_test_loss, result_train_er, result_test_er, sd_list,
                w1_diff_norms, w2_diff_norms, b1_diff_norms, b2_diff_norms, flat_diff_norms, NoTs_mlamb_m0)






def train(model, model_name, seed, run_dir, epochs=10, lr=0.001, batch_size=64, opti="SGD"):

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
        # save the model for each epoch
        torch.save(model.state_dict(), run_dir + '/' + model_name + '_epoch_{}.pth'.format(epoch))

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


        tqdm.write("Model {} Epoch {} - Training loss: {}".format(model_name, epoch, running_loss/total))

    # save the model for each epoch
    torch.save(model.state_dict(), run_dir + '/' + model_name + '_epoch_{}.pth'.format(epochs))
        

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

    return model



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
    run_dir = out_dir + '/' + run_name

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    sgd_seeds = np.random.randint(0, 100000, 2)
    print("sgd_seeds:", sgd_seeds)


    print("Initialise two models")
    matcher = NetMatching(device, args.rebasin_by_activation)
    m0 = matcher.m0
    m1 = matcher.m1
    matcher.compare_models(m0, m1, tag="Initial")


    print("Train two models")
    m0 = train(m0, "m0", sgd_seeds[0], run_dir, epochs=args.num_epochs, lr=args.learning_rate, batch_size=args.batch_size, opti=args.optimizer)
    m1 = train(m1, "m1", sgd_seeds[1], run_dir, epochs=args.num_epochs, lr=args.learning_rate, batch_size=args.batch_size, opti=args.optimizer)
    matcher.compare_models(m0, m1, tag="Trained")


    # Load the trained models for each epoch in order to compare the weights
    print("Load the trained models for each epoch, get the training dynamic matrix.")
    lambs = []
    dyn_loss_barrier_train = []
    dyn_loss_barrier_test = []
    dyn_er_barrier_train = []
    dyn_er_barrier_test = []
    dyn_w1_diff_norms = []
    dyn_w2_diff_norms = []
    dyn_b1_diff_norms = []
    dyn_b2_diff_norms = []
    dyn_flat_diff_norms = []
    dyn_NoTs_mlamb_m0 = []

    rb_dyn_loss_barrier_train = []
    rb_dyn_loss_barrier_test = []
    rb_dyn_er_barrier_train = []
    rb_dyn_er_barrier_test = []
    rb_dyn_w1_diff_norms = []
    rb_dyn_w2_diff_norms = []
    rb_dyn_b1_diff_norms = []
    rb_dyn_b2_diff_norms = []
    rb_dyn_flat_diff_norms = []
    rb_dyn_NoTs_mlamb_m0 = []

    m0 = Net(device).to(device)
    m1 = Net(device).to(device)
    for epoch in tqdm(range(args.num_epochs+1), desc="Eval Checkpoints on each epoch", leave=False):
        m0.load_state_dict(torch.load(run_dir + '/m0_epoch_{}.pth'.format(epoch), map_location=device))
        m1.load_state_dict(torch.load(run_dir + '/m1_epoch_{}.pth'.format(epoch), map_location=device))
        if matcher.rebasin_by_activation:
            perm = matcher.closest_permutation_by_activation(m0, m1, matcher.train_loader)
        else:
            perm = matcher.closest_permutation(m0.fc1.weight, m1.fc1.weight)
        m1_rebased = matcher.permute_weights(m1, perm)
        lambs, lmc_train_loss, lmc_test_loss, lmc_train_er, lmc_test_er, \
            w1_diff_norms, w2_diff_norms, b1_diff_norms, b2_diff_norms, flat_diff_norms, NoTs_mlamb_m0 = matcher.compare_models(m0, m1)
        lambs_rb, lmc_train_loss_rb, lmc_test_loss_rb, lmc_train_er_rb, lmc_test_er_rb, \
            w1_diff_norms_rb, w2_diff_norms_rb, b1_diff_norms_rb, b2_diff_norms_rb, flat_diff_norms_rb, NoTs_mlamb_m0_rb = matcher.compare_models(m0, m1_rebased)
        dyn_loss_barrier_train.append(lmc_train_loss)
        dyn_loss_barrier_test.append(lmc_test_loss)
        dyn_er_barrier_train.append(lmc_train_er)
        dyn_er_barrier_test.append(lmc_test_er)
        dyn_w1_diff_norms.append(w1_diff_norms)
        dyn_w2_diff_norms.append(w2_diff_norms)
        dyn_b1_diff_norms.append(b1_diff_norms)
        dyn_b2_diff_norms.append(b2_diff_norms)
        dyn_flat_diff_norms.append(flat_diff_norms)
        dyn_NoTs_mlamb_m0.append(NoTs_mlamb_m0)

        rb_dyn_loss_barrier_train.append(lmc_train_loss_rb)
        rb_dyn_loss_barrier_test.append(lmc_test_loss_rb)
        rb_dyn_er_barrier_train.append(lmc_train_er_rb)
        rb_dyn_er_barrier_test.append(lmc_test_er_rb)
        rb_dyn_w1_diff_norms.append(w1_diff_norms_rb)
        rb_dyn_w2_diff_norms.append(w2_diff_norms_rb)
        rb_dyn_b1_diff_norms.append(b1_diff_norms_rb)
        rb_dyn_b2_diff_norms.append(b2_diff_norms_rb)
        rb_dyn_flat_diff_norms.append(flat_diff_norms_rb)
        rb_dyn_NoTs_mlamb_m0.append(NoTs_mlamb_m0_rb)


    # list to np array
    dyn_loss_barrier_train = np.array(dyn_loss_barrier_train)
    dyn_loss_barrier_test = np.array(dyn_loss_barrier_test)
    dyn_er_barrier_train = np.array(dyn_er_barrier_train)
    dyn_er_barrier_test = np.array(dyn_er_barrier_test)
    dyn_w1_diff_norms = np.array(dyn_w1_diff_norms)
    dyn_w2_diff_norms = np.array(dyn_w2_diff_norms)
    dyn_b1_diff_norms = np.array(dyn_b1_diff_norms)
    dyn_b2_diff_norms = np.array(dyn_b2_diff_norms)
    dyn_flat_diff_norms = np.array(dyn_flat_diff_norms)
    dyn_NoTs_mlamb_m0 = np.array(dyn_NoTs_mlamb_m0)

    rb_dyn_loss_barrier_train = np.array(rb_dyn_loss_barrier_train)
    rb_dyn_loss_barrier_test = np.array(rb_dyn_loss_barrier_test)
    rb_dyn_er_barrier_train = np.array(rb_dyn_er_barrier_train)
    rb_dyn_er_barrier_test = np.array(rb_dyn_er_barrier_test)
    rb_dyn_w1_diff_norms = np.array(rb_dyn_w1_diff_norms)
    rb_dyn_w2_diff_norms = np.array(rb_dyn_w2_diff_norms)
    rb_dyn_b1_diff_norms = np.array(rb_dyn_b1_diff_norms)
    rb_dyn_b2_diff_norms = np.array(rb_dyn_b2_diff_norms)
    rb_dyn_flat_diff_norms = np.array(rb_dyn_flat_diff_norms)
    rb_dyn_NoTs_mlamb_m0 = np.array(rb_dyn_NoTs_mlamb_m0)

    # save the results
    results = {
            "lambs": lambs,
            "epochs": np.arange(args.num_epochs+1),
            "dyn_loss_barrier_train": dyn_loss_barrier_train,
            "dyn_loss_barrier_test": dyn_loss_barrier_test,
            "dyn_er_barrier_train": dyn_er_barrier_train,
            "dyn_er_barrier_test": dyn_er_barrier_test,
            "dyn_w1_diff_norms": dyn_w1_diff_norms,
            "dyn_w2_diff_norms": dyn_w2_diff_norms,
            "dyn_b1_diff_norms": dyn_b1_diff_norms,
            "dyn_b2_diff_norms": dyn_b2_diff_norms,
            "dyn_flat_diff_norms": dyn_flat_diff_norms,
            "dyn_NoTs_mlamb_m0": dyn_NoTs_mlamb_m0,

            "rb_dyn_loss_barrier_train": rb_dyn_loss_barrier_train,
            "rb_dyn_loss_barrier_test": rb_dyn_loss_barrier_test,
            "rb_dyn_er_barrier_train": rb_dyn_er_barrier_train,
            "rb_dyn_er_barrier_test": rb_dyn_er_barrier_test,
            "rb_dyn_w1_diff_norms": rb_dyn_w1_diff_norms,
            "rb_dyn_w2_diff_norms": rb_dyn_w2_diff_norms,
            "rb_dyn_b1_diff_norms": rb_dyn_b1_diff_norms,
            "rb_dyn_b2_diff_norms": rb_dyn_b2_diff_norms,
            "rb_dyn_flat_diff_norms": rb_dyn_flat_diff_norms,
            "rb_dyn_NoTs_mlamb_m0": rb_dyn_NoTs_mlamb_m0,
        }
    

    utils_runs.save_train_results(out_dir, results, run_name, "training_dynamic")
