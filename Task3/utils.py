import csv
import random
import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import torch
from sklearn.metrics import roc_curve
from torch.utils.data import Dataset


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
    plt.savefig(f"{name}.png", bbox_inches="tight")


def plot_losses(valid_losses, train_losses, ylabel="", name=""):
    plt.clf()
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    # plt.xticks(epochs)
    plt.plot(valid_losses, label="validation")
    plt.plot(train_losses, label="train")
    plt.legend()
    plt.savefig(f"{name}.png", bbox_inches="tight")


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


def train_batch(X, y, model, optimizer, criterion, gpu_id=None, **kwargs):
    """
    X (batch_size, 1000, 3): batch of examples
    y (batch_size, 5): ground truth labels_train
    model: Pytorch model
    optimizer: optimizer for the gradient step
    criterion: loss function
    """
    X, y = X.to(gpu_id), y.to(gpu_id)
    optimizer.zero_grad()
    out = model(X, **kwargs)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def predict(model, X, thr):
    """
    Make labels_train predictions for "X" (batch_size, 1000, 3)
    """
    logits_ = model(X)  # (batch_size, n_classes)
    probabilities = torch.sigmoid(logits_).cpu()

    if thr is None:
        return probabilities
    else:
        return np.array(probabilities.numpy() >= thr, dtype=float)


def evaluate(model, dataloader, thr, gpu_id=None):
    """
    model: Pytorch model
    X (batch_size, 1000, 3) : batch of examples
    y (batch_size, 5): ground truth labels_train
    """
    model.eval()  # set dropout and batch normalization layers to evaluation mode
    with torch.no_grad():
        matrix = np.zeros((5, 4))
        for i, (x_batch, y_batch) in enumerate(dataloader):
            print('eval {} of {}'.format(i + 1, len(dataloader)), end='\r')
            x_batch, y_batch = x_batch.to(gpu_id), y_batch.to(gpu_id)
            y_pred = predict(model, x_batch, thr)
            y_true = np.array(y_batch.cpu())
            matrix = compute_scores(y_true, y_pred, matrix)

            del x_batch
            del y_batch
            torch.cuda.empty_cache()

        model.train()

    return matrix
    # cols: TP, FN, FP, TN


def evaluate_with_norm(model, dataloader, thr, gpu_id=None):
    """
    model: Pytorch model
    X (batch_size, 1000, 3) : batch of examples
    y (batch_size, 5): ground truth labels_train
    """
    model.eval()
    with torch.no_grad():
        matrix = np.zeros((5, 4))
        norm_vec = np.zeros(4)
        for i, (x_batch, y_batch) in enumerate(dataloader):
            # print('eval {} of {}'.format(i + 1, len(dataloader)), end='\r')
            x_batch, y_batch = x_batch.to(gpu_id), y_batch.to(gpu_id)
            y_pred = predict(model, x_batch, thr)
            y_true = np.array(y_batch.cpu())
            # print(y_true)
            # print(y_pred)
            # print()
            matrix, norm_vec = compute_scores_with_norm(y_true, y_pred, matrix, norm_vec)

            del x_batch
            del y_batch
            torch.cuda.empty_cache()

        model.train()

    return matrix, norm_vec
    # cols: TP, FN, FP, TN


def auroc(model, dataloader, gpu_id=None):
    """
    model: Pytorch model
    X (batch_size, 1000, 3) : batch of examples
    y (batch_size, 5): ground truth labels_train
    """
    model.eval()  # set dropout and batch normalization layers to evaluation mode
    with torch.no_grad():
        preds = []
        trues = []
        for i, (x_batch, y_batch) in enumerate(dataloader):
            # print('eval {} of {}'.format(i + 1, len(dataloader)), end='\r')
            x_batch, y_batch = x_batch.to(gpu_id), y_batch.to(gpu_id)

            preds += predict(model, x_batch, None)
            trues += [y_batch.cpu()[0]]

            del x_batch
            del y_batch
            torch.cuda.empty_cache()

    preds = torch.stack(preds)
    trues = torch.stack(trues).int()
    return MultilabelAUROC(num_labels=5, average=None)(preds, trues)
    # cols: TP, FN, FP, TN


# Validation loss
def compute_loss(model, dataloader, criterion, gpu_id=None):
    model.eval()
    with torch.no_grad():
        val_losses = []
        for i, (x_batch, y_batch) in enumerate(dataloader):
            # print('eval {} of {}'.format(i + 1, len(dataloader)), end='\r')
            x_batch, y_batch = x_batch.to(gpu_id), y_batch.to(gpu_id)
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            val_losses.append(loss.item())
            del x_batch
            del y_batch
            torch.cuda.empty_cache()

        model.train()

        return statistics.mean(val_losses)


def threshold_optimization(model, dataloader, gpu_id=None):
    """
    Make labels_train predictions for "X" (batch_size, 1000, 3)
    """
    save_probs = []
    save_y = []
    threshold_opt = np.zeros(5)

    model.eval()
    with torch.no_grad():
        #threshold_opt = np.zeros(5)
        for _, (X, Y) in enumerate(dataloader):
            X, Y = X.to(gpu_id), Y.to(gpu_id)

            Y = np.array(Y.cpu())
            #print(Y)

            logits_ = model(X)  # (batch_size, n_classes)
            probabilities = torch.sigmoid(logits_).cpu()

            save_probs += [probabilities.numpy()]
            save_y += [Y]

    # find the optimal threshold with ROC curve for each disease

    save_probs = np.array(np.concatenate(save_probs)).reshape((-1, 5))
    save_y = np.array(np.concatenate(save_y)).reshape((-1, 5))
    for dis in range(0, 5):
        # print(probabilities[:, dis])
        # print(Y[:, dis])
        fpr, tpr, thresholds = roc_curve(save_y[:, dis], save_probs[:, dis])
        # geometric mean of sensitivity and specificity
        gmean = np.sqrt(tpr * (1 - fpr))
        # optimal threshold
        index = np.argmax(gmean)
        threshold_opt[dis] = round(thresholds[index], ndigits=2)

    return threshold_opt

# performance evaluation, compute the tp, fn, fp, and tp for each disease class
# and compute the specificity and sensitivity
def compute_scores(y_true, y_pred, matrix):
    for j in range(len(y_true)):
        pred = y_pred[j]
        gt = y_true[j]
        for i in range(0, 5):  # for each class
            matrix = computetpfnfp(pred[i], gt[i], i, matrix)
    return matrix


def compute_scores_with_norm(y_true, y_pred, matrix, norm_vec):
    for j in range(len(y_true)):
        pred = y_pred[j]
        gt = y_true[j]
        norm_pred = True
        norm_gt = True
        for i in range(0, 5):  # for each class
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

def compute_metrics(matrix, class_names=None, save_as=None):
    n = matrix.shape[0]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n)]
    assert len(class_names) == n, "Length of class_names must match matrix dimensions"

    # matrix shape: (n_classes, 4) — columns = [TP, FP, FN, TN]
    TP = matrix[:, 0]
    FN = matrix[:, 1]
    FP = matrix[:, 2]
    TN = matrix[:, 3]

    # Avoid division by zero with np.where
    sensitivity = np.where((TP + FN) > 0, TP / (TP + FN), 0)
    specificity = np.where((TN + FP) > 0, TN / (TN + FP), 0)
    accuracy    = np.where((TP + FP + FN + TN) > 0, (TP + TN) / (TP + FP + FN + TN), 0)
    precision   = np.where((TP + FP) > 0, TP / (TP + FP), 0)
    f1          = np.where((precision + sensitivity) > 0, 2 * precision * sensitivity / (precision + sensitivity), 0)

    # Create DataFrame
    data = {
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Accuracy": accuracy,
        "Precision": precision,
        "F1 Score": f1
    }
    df = pd.DataFrame(data, index=class_names)

    # Append Mean row
    mean_row = df.mean(axis=0)
    mean_row.name = "Mean"
    df = pd.concat([df, pd.DataFrame([mean_row])])

    # Round values to 2 decimal places
    df = df.round(2)

    # Save to CSV if specified
    if save_as is not None:
        df.to_csv(f"{save_as}.csv")

    return df

def compute_save_metrics(matrix, matrix_dev, opt_threshold, date, epoch, strategy, path_save_model, learning_rate,
                         optimizer, dropout, epochs, hidden_size, batch_size, test_id):
    classes = ["AFIB", "AFLT", "1dAVb", "RBBB", "LBBB"]
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
