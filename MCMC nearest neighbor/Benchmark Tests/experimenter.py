import sys
import os
import time
import datetime as dt
import pandas as pd
import numpy as np
from tqdm import tqdm


# plotting imports
import matplotlib.pyplot as plt

# ml imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# random forest
from sklearn.ensemble import RandomForestClassifier
# k nearest neighbors
from sklearn.neighbors import KNeighborsClassifier

# metric imports

# our model imports
sys.path.append('../')
from model import model3



### GLOBAL VARIABLES ###

benchmarking = {}
repeat = 5
info_length = 3     # train_acc, test_acc, time
date_format = "%y%m%d@%H%M"
possible_models = ["randomforest", "knn", "cnn", "ours", "metric"]





### HELPER FUNCTIONS ###

def get_last_file(dir:str, partial_name:str):
    """
    Get the last file in a directory matching the partial name

    Parameters:
    partial_name (str): partial name of the file
    dir (str): directory to search in

    Returns:
    str: path to the last file
    """
    # get all the datetimes
    datetimes = [name.split("_")[-1].split(".")[0][1:] for name in os.listdir(dir) if partial_name in name]
    
    # get the latest datetime by converting to datetime object
    vals = [dt.datetime.strptime(date, date_format) for date in datetimes]
    if len(vals) == 0:
        return None
    
    # get the latest datetime and return the path
    latest_datetime = max(vals).strftime(date_format)
    return os.path.join(dir, f"{partial_name}_d{latest_datetime}.npy")


def build_dir(dataset_name:str, model_name:str):
    """
    Build the directory to save the numpy files
    Parameters:
        dataset_name (str): name of the dataset
        model_name (str): name of the model
    Returns:
        str: directory
    """
    return f"results/{dataset_name}/{model_name}/npy_files"


def build_name(val:bool, info_type:str, iteration):
    """
    Build the name of the numpy file
    Parameters:
        val (bool): whether the data is validation data
        info_type (str): type of information
        iteration (int): iteration number
    Returns:
        str: name of the numpy file
    """
    train = "train" if not val else "val"
    return f"{train}_{info_type}_i{iteration}"


def load_data(dataset_name:str, model_name:str, info_type:str, iteration, val=False):
    """
    Load the data from the most recent numpy file matching the partial name
    Parameters:
        dataset_name (str): name of the dataset
        model_name (str): name of the model
        info_type (str): type of information
        iteration (int): iteration number
        val (bool): whether the data is validation data
    Returns:
        np.ndarray: data from the numpy file or None if the file does not exist
    """
    dir = build_dir(dataset_name, model_name)
    partial_name = build_name(val, info_type, iteration)
    try:
        return np.load(get_last_file(dir, partial_name))
    except:
        print("Error: file not found")
        return None
    

def save_data(data:np.ndarray, save_constants:tuple, info_type:str, iteration, 
              val=False, refresh=False):
    """
    Save the data to a numpy file
    Parameters:
        data (np.ndarray): data to save
        save_constants (tuple): contains
            dataset_name (str): name of the dataset
            model_name (str): name of the model
            datetime (str): current date and time
        info_type (str): type of information
        iteration (int): iteration number
        val (bool): whether the data is validation data
        refresh (bool): whether to refresh the last file
    Returns: None
    """
    dataset_name, model_name, datetime = save_constants
    dir = build_dir(dataset_name, model_name)
    partial_name = build_name(val, info_type, iteration)
    if not os.path.exists(dir):
        os.makedirs(dir)
        print("Made a new directory", dir)
    
    # check if the directory has files
    if len(os.listdir(dir)) < info_length * repeat:
        refresh = False

    # remove the last file with the same partial name
    if refresh:
        last_file = get_last_file(dir, partial_name)
        if last_file is not None:
            os.remove(last_file)

    path = f"results/{dataset_name}/{model_name}/npy_files/{partial_name}_d{datetime}.npy"
    np.save(path, data)
    return None



### MODEL FUNCTIONS ###

def benchmark_ml(model_name, experiment_info, datetime, 
                save_all=True, save_any=True, refresh=True):
    """
    Trains a model on the cancer dataset with different data sizes and saves the accuracy and time data.

    Parameters:
        model_name (str): The name of the model to train. Can be "randomforest", "knn", or "ours".
        experiment_info (tuple): Contains
            dataset_name (str): The name of the dataset to train on.
            dataset_sizes (list(int)): A list of the sizes of the dataset to train on.
            X_train (np.ndarray): The training data.
            y_train (np.ndarray): The training labels.
            X_test (np.ndarray): The testing data.
            y_test (np.ndarray): The testing labels.
        datetime (str): The current date and time.
        save_all (bool): Whether to save all the data or just the means and stds
        save_any (bool): Whether to save any data at all
        refresh (bool): Whether to refresh the last file

    Returns:
    results_dict (dict): A dictionary containing the accuracy and time data for each model and iteration
    """
    if not save_any:
        save_all = False
    # unpack experiment info
    dataset_name, data_sizes, X_train, y_train, X_test, y_test = experiment_info
    model = RandomForestClassifier(n_jobs=-1) if model_name == "randomforest" else \
            KNeighborsClassifier(n_jobs=-1) if model_name == "knn" else \
            model3() if model_name == "ours" else None
    results_dict = {model_name: {}}
    save_constants = (dataset_name, model_name, datetime)
    
    progressB = tqdm(total=repeat*len(data_sizes), desc=f"Benchmarking {model_name}")
    for i in range(repeat):
        time_list = []
        train_acc = []
        val_acc = []

        for size in data_sizes:
            if size is None or size > len(X_train):
                data_sizes[data_sizes.index(size)] = len(X_train)
                size = len(X_train)
            # train model
            clf = model
            start_time = time.perf_counter()
            clf.fit(X_train[:size], y_train[:size])
            end_time = time.perf_counter()
            y_pred = clf.predict(X_test)

            # predict and compute accuracy
            y_pred_train = clf.predict(X_train[:size])
            y_pred = clf.predict(X_test)
            acc_train = accuracy_score(y_train[:size], y_pred_train)
            acc_test = accuracy_score(y_test, y_pred)

            # update lists
            time_list.append(end_time - start_time)
            train_acc.append(acc_train)
            val_acc.append(acc_test)
            progressB.update(1)

        # Done training, now saving data
        train_acc = np.array(train_acc)
        val_acc = np.array(val_acc)
        time_list = np.array(time_list)
        if save_all:
            save_data(train_acc, save_constants, "acc", i, refresh=refresh)
            save_data(val_acc, save_constants, "acc", i, val=True, refresh=refresh)
            save_data(time_list, save_constants, "time", i, refresh=refresh)
        results_dict[model_name][i] = {"train_acc": train_acc, "val_acc": val_acc, "time": time_list}
    progressB.close()


    # Done benchmarking, calculate means and stds and saving them
    train_accs = np.array([results_dict[model_name][i]["train_acc"] for i in range(repeat)])
    val_accs = np.array([results_dict[model_name][i]["val_acc"] for i in range(repeat)])
    times = np.array([results_dict[model_name][i]["time"] for i in range(repeat)])
    train_acc_mean = np.mean(train_accs, axis=0)
    val_acc_mean = np.mean(val_accs, axis=0)
    time_mean = np.mean(times, axis=0)
    train_acc_std = np.std(train_accs, axis=0)
    val_acc_std = np.std(val_accs, axis=0)
    time_std = np.std(times, axis=0)

    # save means and stds
    if not save_any:
        save_data(train_acc_mean, save_constants,  "acc", "mean", refresh=refresh)
        save_data(val_acc_mean, save_constants, "acc", "mean", val=True, refresh=refresh)
        save_data(time_mean, save_constants, "time", "mean", refresh=refresh)
        save_data(train_acc_std, save_constants, "acc", "std", refresh=refresh)
        save_data(val_acc_std, save_constants, "acc", "std", val=True, refresh=refresh)
        save_data(time_std, save_constants, "time", "std", refresh=refresh)
    results_dict[model_name]["mean"] = {"train_acc": train_acc_mean, "val_acc": val_acc_mean, "time": time_mean}
    results_dict[model_name]["std"] = {"train_acc": train_acc_std, "val_acc": val_acc_std, "time": time_std}
    
    return results_dict





### MODEL EVALUATION ###

def rebuild_results(benchmarking, all_data=False):
    """
    Rebuild the benchmarking results from the numpy files
    Parameters:
        benchmarking (dict): dictionary to store the results
        all_data (bool): whether to load all the data or just the means and std
    Returns:
        None
    """
    for model_name in possible_models:
        # skip checking
        if model_name in benchmarking.keys():
            continue
        folder = build_dir(model_name)
        if len(os.listdir(folder)) == 0:
            continue

        benchmarking[model_name] = {}

        if all_data:
            for i in range(repeat):
                benchmarking[model_name][i] = {"train_acc": load_data(model_name, "acc", i),
                                               "val_acc": load_data(model_name, "acc", i, val=True),
                                               "time": load_data(model_name, "time", i)}
        benchmarking[model_name]["mean"] = {"train_acc": load_data(model_name, "acc", "mean"),
                                            "val_acc": load_data(model_name, "acc", "mean", val=True),
                                            "time": load_data(model_name, "time", "mean")}
        benchmarking[model_name]["std"] = {"train_acc": load_data(model_name, "acc", "std"),
                                        "val_acc": load_data(model_name, "acc", "std", val=True),
                                        "time": load_data(model_name, "time", "std")}
    return benchmarking


def plot_results(benchmarking, constants, scale=5, save_fig=True, from_data=True):
    """
    Plot the benchmarking results

    Parameters:
        benchmarking (dict): dictionary containing the benchmarking results
        constants (tuple): contains
            data_sizes (list(int)): list of data sizes
            datetime (str): current date and time
            dataset_name (str): name of the dataset
        scale (int): scale of the figure
        save_fig (bool): whether to save the figure
        from_data (bool): whether to load the data from the numpy files
    
    Returns:
        None
    """
    data_sizes, datetime, dataset_name = constants
    info_ylabels = ["Accuracy (%)", "Accuracy (%)", "Training Time (s)"]
    colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e']

    if from_data:
        benchmarking = rebuild_results(benchmarking)

    plt.figure(figsize=(scale*info_length, scale), dpi=150)
    for j, (model_name, model_results) in enumerate(benchmarking.items()):
        for i, (info_type, means), (_, stds) in zip(range(info_length), model_results["mean"].items(), model_results["std"].items()):
            plt.subplot(1, info_length, i+1)
            plt.plot(data_sizes, means, label=model_name, marker='o', color=colors[j])
            plt.fill_between(data_sizes, means - stds, means + stds, alpha=0.2, color=colors[j])
            plt.xlabel("Data Size")
            plt.ylabel(info_ylabels[i])
            if "acc" in info_type:
                plt.axhline(y=1, color='k', linestyle='--')
            if "time" in info_type:
                plt.yscale('log')
            plt.title(info_type)
            plt.legend()
    plt.suptitle("Model Benchmarking")
    plt.tight_layout()
    if save_fig:
        plt.savefig(f"results/{dataset_name}/charts/benchmarking_{datetime}.png")
    plt.show()