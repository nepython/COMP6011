# code based on the source code of homework 1 and homework 2 of the
# deep structured learning code https://fenix.tecnico.ulisboa.pt/disciplinas/AEProf/2021-2022/1-semestre/homeworks

# import the necessary packages
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import csv
import tifffile


def configure_device(gpu_id):
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)


def configure_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def plot(plottable, ylabel="", name=""):
    plt.clf()
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.plot(plottable)
    plt.savefig("%s.pdf" % (name), bbox_inches="tight")


def plot_losses(epochs, valid_losses, train_losses, ylabel="", name=""):
    plt.clf()
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    # plt.xticks(epochs)
    plt.plot(valid_losses, label="validation")
    plt.plot(train_losses, label="train")
    plt.legend()
    plt.savefig("%s.pdf" % (name), bbox_inches="tight")


# create a generator to read the images as we train the model
# (similar to flow_from_directory Keras)
class ECGImageDataset(Dataset):
    """
    path/train/images
              /labels
        /val/images
            /labels
        /test/images
             /labels
    """

    def __init__(self, path, train_dev_test, part="train"):
        self.path = path
        self.part = part
        self.train_dev_test = train_dev_test

    def __len__(self):
        if self.part == "train":
            return self.train_dev_test[0]
        elif self.part == "dev":
            return self.train_dev_test[1]
        elif self.part == "test":
            return self.train_dev_test[2]

    def __getitem__(self, idx):
        X, y = read_data_for_CNN(self.path, self.part, idx)
        return torch.tensor(X).float(), torch.tensor(y).float()


def read_data_for_CNN(path, partition, idx):
    """Read the ECG Image Data"""
    path_labels = str(path) + "labels_" + str(partition)
    path_X = str(path) + "X_cnn_" + str(partition)
    index = idx
    label = np.load(str(path_labels) + "/" + str(index) + ".npy")
    image = tifffile.imread(str(path_X) + "/" + str(index) + ".tif")
    image = image / 255.0  # normalization
    return image, label


class Dataset_for_RNN(Dataset):
    """
    path/labels_train
        /X_train
        /labels_val
        /X_val
        /labels_test
        /X_test
    """

    def __init__(self, path, train_dev_test, part="train"):
        self.path = path
        self.part = part
        self.train_dev_test = train_dev_test

    def __len__(self):
        if self.part == "train":
            return self.train_dev_test[0]
        elif self.part == "dev":
            return self.train_dev_test[1]
        elif self.part == "test":
            return self.train_dev_test[2]

    def __getitem__(self, idx):
        X, y = read_data_for_RNN(self.path, self.part, idx)
        return torch.tensor(X).float(), torch.tensor(y).float()


def read_data_for_RNN(path, partition, idx):
    path_labels = str(path) + "labels_" + str(partition)
    path_X = str(path) + "X_rnn_" + str(partition)
    index = idx
    label = np.load(str(path_labels) + "/" + str(index) + ".npy")
    X = np.load(str(path_X) + "/" + str(index) + ".npy")
    return X, label


# performance evaluation, compute the tp, fn, fp, and tp for each disease class
# and compute the specificity and sensitivity
def compute_scores(y_true, y_pred, matrix):
    for j in range(len(y_true)):
        pred = y_pred[j]
        gt = y_true[j]
        for i in range(0, 4):  # for each class
            matrix = computetpfnfp(pred[i], gt[i], i, matrix)
    return matrix


def compute_scores_with_norm(y_true, y_pred, matrix, norm_vec):
    for j in range(len(y_true)):
        pred = y_pred[j]
        gt = y_true[j]
        norm_pred = True
        norm_gt = True
        for i in range(0, 4):  # for each class
            matrix = computetpfnfp(pred[i], gt[i], i, matrix)
            if gt[i] == 1 & norm_gt:
                norm_gt = False
            if pred[i] == 1 & norm_pred:
                norm_pred = False
        if norm_gt == 0 and norm_pred == 0:  # tn
            norm_vec[3] += 1
        if norm_gt == 1 and norm_pred == 0:  # fn
            norm_vec[1] += 1
        if norm_gt == 0 and norm_pred == 1:  # fp
            norm_vec[2] += 1
        if norm_gt == 1 and norm_pred == 1:  # tp
            norm_vec[0] += 1
    return matrix, norm_vec


def compute_scores_dev(matrix):
    matrix[matrix == 0] = 0.01
    # print(matrix)
    sensitivity = matrix[:, 0] / (matrix[:, 0] + matrix[:, 1])  # tp/(tp+fn)
    specificity = matrix[:, 3] / (matrix[:, 3] + matrix[:, 2])  # tn/(tn+fp)
    return np.mean(sensitivity), np.mean(specificity)


def computetpfnfp(pred, gt, i, matrix):
    if gt == 0 and pred == 0:  # tn
        matrix[i, 3] += 1
    if gt == 1 and pred == 0:  # fn
        matrix[i, 1] += 1
    if gt == 0 and pred == 1:  # fp
        matrix[i, 2] += 1
    if gt == 1 and pred == 1:  # tp
        matrix[i, 0] += 1
    return matrix


def compute_save_metrics(matrix, matrix_dev, opt_threshold, date, epoch, strategy, path_save_model, learning_rate,
                         optimizer, dropout, epochs, hidden_size, batch_size, test_id):
    classes = ["NORM", "AFIB", "AFLT", "1dAVb", "RBBB"]
    sensitivities = []
    specificities = []

    # Compute sensitivity and specificity for each class
    for i in range(len(classes)):
        tp, fp, fn, tn = matrix[i]
        sensi = tp / (tp + fn)
        spec = tn / (tn + fp)
        sensitivities.append(sensi)
        specificities.append(spec)

    # Compute mean sensitivity and specificity
    mean_sensi = np.mean(matrix[:, 0]) / (np.mean(matrix[:, 0]) + np.mean(matrix[:, 2]))
    mean_spec = np.mean(matrix[:, 3]) / (np.mean(matrix[:, 3]) + np.mean(matrix[:, 1]))
    mean_sensi_dev = np.mean(matrix_dev[:, 0]) / (np.mean(matrix_dev[:, 0]) + np.mean(matrix_dev[:, 1]))
    mean_spec_dev = np.mean(matrix_dev[:, 3]) / (np.mean(matrix_dev[:, 3]) + np.mean(matrix_dev[:, 2]))

    # Print results
    print("Final Test Results:")
    for i, cls in enumerate(classes):
        print(f"{cls}: sensitivity - {sensitivities[i]:.2f}; specificity - {specificities[i]:.2f}")
    print(f"mean: sensitivity - {mean_sensi:.2f}; specificity - {mean_spec:.2f}")

    # Save to file
    with open(f'results/model/{strategy}.txt', 'w') as f:
        f.write("Final Test Results:\n")
        for i, cls in enumerate(classes):
            f.write(f"{cls}: sensitivity - {sensitivities[i]:.2f}; specificity - {specificities[i]:.2f}\n")
        f.write(f"mean: sensitivity - {mean_sensi:.2f}; specificity - {mean_spec:.2f}\n")
