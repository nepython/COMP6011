# Code based on the source code of homework 1 and homework 2 of the
# deep structured learning code https://fenix.tecnico.ulisboa.pt/disciplinas/AEProf/2021-2022/1-semestre/homeworks

import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils import configure_seed, configure_device, compute_scores, compute_scores_dev, Dataset_for_RNN, \
    plot_losses, plot, compute_scores_with_norm, compute_metrics, train_batch, predict, evaluate, \
    evaluate_with_norm, auroc, compute_loss, threshold_optimization

from datetime import datetime
import numpy as np
import os
from torchmetrics.classification import MultilabelAUROC
from config import samples, class_weight, class_names

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, n_classes, dropout_rate, bidirectional, gpu_id=None,
                 **kwargs):
        """
        Define the layers of the model
        Args:
            input_size (int): "Feature" size (in this case, it is 3)
            hidden_size (int): Number of hidden units
            num_layers (int): Number of hidden GRU layers
            n_classes (int): Number of classes in our classification problem
            dropout_rate (float): Dropout rate to be applied in all rnn layers except the last one
            bidirectional (bool): Boolean value: if true, gru layers are bidirectional
        """
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_classes = n_classes
        self.gpu_id = gpu_id
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional

        # RNN can be replaced with GRU/LSTM (for GRU the rest of the model stays exactly the same)
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout_rate, batch_first=True,
                          bidirectional=bidirectional)  # batch_first: first dimension is the batch size

        if bidirectional:
            self.d = 2
        else:
            self.d = 1

        self.fc = nn.Linear(hidden_size*self.d, n_classes)  # linear layer for the classification part

    def forward(self, X, **kwargs):
        """
        Forward Propagation

        Args:
            X: batch of training examples with dimension (batch_size, 1000, 3)
        """
        # initial hidden state:
        h_0 = torch.zeros(self.num_layers*self.d, X.size(0), self.hidden_size).to(self.gpu_id)

        out_rnn, _ = self.rnn(X.to(self.gpu_id), h_0)
        # out_rnn shape: (batch_size, seq_length, hidden_size*d) = (batch_size, 1000, hidden_size*d)

        if self.bidirectional:
            # concatenate last timestep from the "left-to-right" direction and the first timestep from the
            # "right-to-left" direction
            out_rnn = torch.cat((out_rnn[:, -1, :self.hidden_size], out_rnn[:, 0, self.hidden_size:]), dim=1)
        else:
            # last timestep
            out_rnn = out_rnn[:, -1, :]

        # out_rnn shape: (batch_size, hidden_size*d) - ready to enter the fc layer
        out_fc = self.fc(out_rnn)
        # out_fc shape: (batch_size, num_classes)

        return out_fc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', default='data_for_rnn/',
                        help="Path to the dataset.")
    parser.add_argument('-epochs', default=200, type=int,
                        help="""Number of epochs to train the model.""")
    parser.add_argument('-batch_size', default=512, type=int,
                        help="Size of training batch.")
    parser.add_argument('-learning_rate', type=float, default=0.01)
    parser.add_argument('-dropout', type=float, default=0.3)
    parser.add_argument('-l2_decay', type=float, default=0)
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='adam')
    parser.add_argument('-gpu_id', type=int, default=None)
    parser.add_argument('-path_save_model', default='save_models/',
                        help='Path to save the model')
    parser.add_argument('-num_layers', type=int, default=2)
    parser.add_argument('-hidden_size', type=int, default=256)
    parser.add_argument('-bidirectional', type=bool, default=False)
    parser.add_argument('-early_stop', type=bool, default=True)
    parser.add_argument('-patience', type=int, default=10)
    opt = parser.parse_args()

    configure_seed(seed=42)
    configure_device(opt.gpu_id)

    print("Loading data...")
    train_dataset = Dataset_for_RNN(opt.data, samples, 'train')
    dev_dataset = Dataset_for_RNN(opt.data, samples, 'dev')
    test_dataset = Dataset_for_RNN(opt.data, samples, 'test')

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=opt.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)
    dev_dataloader_thr = DataLoader(dev_dataset, batch_size=1024, shuffle=False)

    input_size = 12
    hidden_size = opt.hidden_size
    num_layers = opt.num_layers
    n_classes = 5

    # initialize the model
    model = GRU(input_size, hidden_size, num_layers, n_classes, dropout_rate=opt.dropout, gpu_id=opt.gpu_id,
                bidirectional=opt.bidirectional)
    model = model.to(opt.gpu_id)

    # get an optimizer
    optims = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD}

    optim_cls = optims[opt.optimizer]
    optimizer = optim_cls(
        model.parameters(),
        lr=opt.learning_rate,
        weight_decay=opt.l2_decay)

    # get a loss criterion and compute the class weights (nbnegative/nbpositive)
    # according to the comments https://discuss.pytorch.org/t/weighted-binary-cross-entropy/51156/6
    # and https://discuss.pytorch.org/t/multi-label-multi-class-class-imbalance/37573/2
    class_weights = torch.tensor(class_weight, dtype=torch.float)
    class_weights = class_weights.to(opt.gpu_id)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=class_weights)  # https://learnopencv.com/multi-label-image-classification-with-pytorch-image-tagging/
    # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html

    # training loop
    epochs = torch.arange(1, opt.epochs + 1)
    train_mean_losses = []
    valid_mean_losses = []
    valid_specificity = []
    valid_sensitivity = []
    train_losses = []
    epochs_run = opt.epochs
    for ii in epochs:
        print('Training epoch {}'.format(ii))
        for i, (X_batch, y_batch) in enumerate(train_dataloader):
            loss = train_batch(
                X_batch, y_batch, model, optimizer, criterion, gpu_id=opt.gpu_id)
            del X_batch
            del y_batch
            torch.cuda.empty_cache()
            train_losses.append(loss)

        mean_loss = torch.tensor(train_losses).mean().item()
        print('Training loss: %.4f' % (mean_loss))

        train_mean_losses.append(mean_loss)
        # Threshold optimization on validation set
        thr = threshold_optimization(model, dev_dataloader_thr)
        matrix = evaluate(model, dev_dataloader, thr, gpu_id=opt.gpu_id)
        sensitivity, specificity = compute_scores_dev(matrix)
        val_loss = compute_loss(model, dev_dataloader, criterion, gpu_id=opt.gpu_id)
        valid_mean_losses.append(val_loss)
        valid_sensitivity.append(sensitivity)
        valid_specificity.append(specificity)
        print('Valid specificity: %.4f' % (valid_specificity[-1]))
        print('Valid sensitivity: %.4f' % (valid_sensitivity[-1]), '\n')

        dt = datetime.now()
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        # save the model at each epoch where the validation loss is the best so far
        if sensitivity == np.max(valid_sensitivity):
            f = os.path.join(opt.path_save_model, model.__class__.__name__ + '_ep_'+ str(ii.item()))
            best_model = ii
            torch.save(model.state_dict(), f)

        # early stop - if validation loss does not increase for 15 epochs, stop learning process
        if opt.early_stop:
            if ii > opt.patience:
                if valid_mean_losses[ii - opt.patience] == np.min(valid_mean_losses[ii - opt.patience:]):
                    epochs_run = ii
                    break

    # Make predictions based on best model (lowest validation loss)
    # Load model
    model.load_state_dict(torch.load(f))
    model.eval()

    # Threshold optimization on validation set
    thr = threshold_optimization(model, dev_dataloader_thr)

    # Results on test set:
    matrix = evaluate(model, test_dataloader, thr, gpu_id=opt.gpu_id)
    print(matrix)
    metrics = compute_metrics(matrix, class_names=class_names, save_as=f'results/{model.__class__.__name__}')
    print(metrics)

    # plot
    plot_losses(valid_mean_losses, train_mean_losses, ylabel='Loss', name=f'results/figures/{model.__class__.__name__}_loss')
    plot(valid_specificity, ylabel='Specificity',
         name=f'results/figures/{model.__class__.__name__}_val_specificity')
    plot(valid_sensitivity, ylabel='Sensitivity',
         name=f'results/figures/{model.__class__.__name__}_val_sensitivity')

if __name__ == '__main__':
    main()
