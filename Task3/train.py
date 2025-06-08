import argparse
import os
from datetime import datetime

import numpy as np
import torch
from config import class_names, class_weight, samples
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MultilabelAUROC

from models.oned import GRU, LSTM, RNN
from models.twod import CNN
from models.twod import VGG16 as VGG
from models.twod import AlexNet
from models.twod import ResNet50 as ResNet
from utils import (Dataset_for_RNN, ECGImageDataset, auroc, compute_loss,
                   compute_metrics, compute_scores, compute_scores_dev,
                   configure_device, configure_seed, evaluate, evaluate_with_norm,
                   plot, plot_losses, predict, threshold_optimization, train_batch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', default='data_for_rnn/',
                        help="Path to the dataset.")
    parser.add_argument('-path_save_model', default='save_models/',
                        help='Path to save the model')
    parser.add_argument('-model', type=str, default='gru',
                        choices=['rnn', 'lstm', 'gru', 'cnn', 'alexnet',
                        'vgg', 'resnet', 'early', 'late', 'joint'],
                        help="Model to train.")
    parser.add_argument('-datatype', type=str, default='rnn',
                        choices=['rnn', 'cnn', 'fusion'],
                        help="Type of model to train.")
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
    parser.add_argument('-num_layers', type=int, default=2)
    parser.add_argument('-hidden_size', type=int, default=256)
    parser.add_argument('-bidirectional', type=bool, default=False)
    parser.add_argument('-early_stop', type=bool, default=True)
    parser.add_argument('-patience', type=int, default=10)
    opt = parser.parse_args()

    configure_seed(seed=42)
    configure_device(opt.gpu_id)

    print("Loading data...")
    if opt.datatype == 'rnn':
        train_dataset = Dataset_for_RNN(opt.data, samples, 'train')
        dev_dataset = Dataset_for_RNN(opt.data, samples, 'dev')
        test_dataset = Dataset_for_RNN(opt.data, samples, 'test')
    elif opt.datatype == 'cnn':
        train_dataset = ECGImageDataset(opt.data, samples, 'train')
        dev_dataset = ECGImageDataset(opt.data, samples, 'dev')
        test_dataset = ECGImageDataset(opt.data, samples, 'test')

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=opt.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)
    dev_dataloader_thr = DataLoader(dev_dataset, batch_size=opt.batch_size, shuffle=False)

    input_size = 12
    hidden_size = opt.hidden_size
    num_layers = opt.num_layers
    n_classes = 5

    # initialize the model
    if opt.model == 'rnn':
        model = RNN(input_size, hidden_size, num_layers, n_classes, dropout_rate=opt.dropout, gpu_id=opt.gpu_id)
    elif opt.model == 'lstm':
        model = LSTM(input_size, hidden_size, num_layers, n_classes, dropout_rate=opt.dropout, gpu_id=opt.gpu_id,
                    bidirectional=opt.bidirectional)
    elif opt.model == 'gru':
        model = GRU(input_size, hidden_size, num_layers, n_classes, dropout_rate=opt.dropout, gpu_id=opt.gpu_id,
                    bidirectional=opt.bidirectional)
    elif opt.model == 'cnn':
        model = CNN(n_classes)
    elif opt.model == 'alexnet':
        model = AlexNet(n_classes)
    elif opt.model == 'vgg':
        model = VGG(n_classes)
    elif opt.model == 'resnet':
        model = ResNet(n_classes)
    model.to(opt.gpu_id)

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
        thr = threshold_optimization(model, dev_dataloader_thr, gpu_id=opt.gpu_id)
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
    thr = threshold_optimization(model, dev_dataloader_thr, gpu_id=opt.gpu_id)

    # Results on test set:
    matrix, norm_vec = evaluate_with_norm(model, test_dataloader, thr, gpu_id=opt.gpu_id)
    matrix = np.vstack([matrix, norm_vec])
    # matrix = evaluate(model, test_dataloader, thr, gpu_id=None)
    # aurocs = auroc(model, test_dataloader, gpu_id=gpu_id)

    print(matrix)
    metrics = compute_metrics(matrix, class_names=class_names+['NORM'], save_as=f'results/{model.__class__.__name__}')
    print(metrics)

    # plot
    plot_losses(valid_mean_losses, train_mean_losses, ylabel='Loss', name=f'results/figures/{model.__class__.__name__}_loss')
    plot(valid_specificity, ylabel='Specificity',
         name=f'results/figures/{model.__class__.__name__}_val_specificity')
    plot(valid_sensitivity, ylabel='Sensitivity',
         name=f'results/figures/{model.__class__.__name__}_val_sensitivity')

if __name__ == '__main__':
    main()
