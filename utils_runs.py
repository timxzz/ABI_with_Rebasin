import os
import torch
import json
import pickle


def save_model(out_dir, run_name, model):
    # Save model
    print('Saving model')
    run_dir = os.path.join(out_dir, run_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    torch.save(model.state_dict(), os.path.join(run_dir, 'vae.pth'))
    print('Saving done!')


def load_model(out_dir, run_name, model, device):
    # Load model
    print('Loading model')
    run_dir = os.path.join(out_dir, run_name)
    model.load_state_dict(torch.load(os.path.join(run_dir, 'vae.pth'), map_location=device))
    print('Loading done!')
    return model


def save_train_results(out_dir, train_results, run_name, save_name):
    # Save evaluation results
    file_dir = os.path.join(out_dir, run_name, save_name+".pkl")
    print("Saving training results to: " + file_dir)
    with open(file_dir, 'wb') as f:
        pickle.dump(train_results, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_train_config(out_dir, run_name, config):
    # Save training config
    run_dir = os.path.join(out_dir, run_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    file_dir = os.path.join(run_dir, "config.json")
    print("Saving config to: " + file_dir)
    with open(file_dir, 'w') as f:
        json.dump(config, f, indent=2)


def save_train_results_as_json(out_dir, run_name, results):
    # Save training results
    run_dir = os.path.join(out_dir, run_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    file_dir = os.path.join(run_dir, "results.json")
    print("Saving results to: " + file_dir)
    with open(file_dir, 'w') as f:
        json.dump(results, f, indent=2)


def get_epoch_num(filename):
    return int(filename.split("_")[2][:-3])

def get_chkpt_list_from_run_dir(run_dir):
    """
    Get a list of config and trained model from the batch of runs
    """
    saved_models_dir = os.path.join(run_dir, 'saved_models')
    print("Searching saved models from: " + saved_models_dir)
    chkpt_list = []
    for subdir, dirs, files in os.walk(saved_models_dir):
        sorted_file_list = sorted(files, key=get_epoch_num)
        for file in sorted_file_list:
            if file.endswith((".pt")):
                chkpt_list.append(os.path.join(saved_models_dir, file))

    return chkpt_list