#Code based on the source code of homework 1 and homework 2 of the 
#deep structured learning code https://fenix.tecnico.ulisboa.pt/disciplinas/AEProf/2021-2022/1-semestre/homeworks

#import packages
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils import configure_seed, configure_device, plot, ECGImageDataset, compute_scores_dev, compute_scores, plot_losses, compute_metrics
#auxiliary functions to evaluate the performance of the model
from sklearn.metrics import recall_score
import statistics
import numpy as np
import os
from config import samples, class_weight, class_names
from sklearn.metrics import roc_curve

# Defined a CNN based on the AlexNet model,
# based on: https://medium.com/analytics-vidhya/alexnet-a-simple-implementation-using-pytorch-30c14e8b6db2 (visited on April 27, 2022)
class AlexNet(nn.Module):
    def __init__(self, n_classes, **kwargs):
        """
        Define the layers of the model
        Args:
            n_classes (int): Number of classes in our classification problem
        """
        super(AlexNet, self).__init__()
        nb_filters = 8 #number of filters in the first layer
        self.n_classes = n_classes
        self.conv2d_1 = nn.Conv2d(36,nb_filters,11,stride=4) #36 input channels
        #nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv2d_2 = nn.Conv2d(nb_filters, nb_filters*2, 5, padding=2)
        self.conv2d_3 = nn.Conv2d(nb_filters*2, nb_filters*4, 3, padding=1)
        self.conv2d_4 = nn.Conv2d(nb_filters*4, nb_filters*8, 3, padding=1)
        self.conv2d_5 = nn.Conv2d(nb_filters*8, 256, 3, padding=1)
        self.linear_1 = nn.Linear(9216, 4096)
        self.linear_2 = nn.Linear(4096, 2048)
        self.linear_3 = nn.Linear(2048, n_classes)
        #nn.MaxPool2d(kernel_size)
        self.maxpool2d = nn.MaxPool2d(3, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0)

    def forward(self, X, **kwargs):
        """
        Forward Propagation
        Args:
            X: batch of training examples with dimension (batch_size, 36, 256, 256) 
        """
        x1 = self.relu(self.conv2d_1(X))
        maxpool1 =  self.maxpool2d(x1)
        maxpool1 = self.dropout(maxpool1)
        x2 = self.relu(self.conv2d_2(maxpool1))
        maxpool2 = self.maxpool2d(x2)
        maxpool2 = self.dropout(maxpool2)
        x3 = self.relu(self.conv2d_3(maxpool2))
        x4 = self.relu(self.conv2d_4(x3))
        x5 = self.relu(self.conv2d_5(x4))
        x6 = self.maxpool2d(x5)
        x6 = self.dropout(x6)
        x6 = x6.reshape(x6.shape[0],-1) #flatten (batch_size,)
        x7 = self.relu(self.linear_1(x6))
        x8 = self.relu(self.linear_2(x7))
        x9 = self.linear_3(x8)
        return x9

def train_batch(X, y, model, optimizer, criterion, gpu_id=None, **kwargs):
    """
    X (batch_size, 9, 256, 256): batch of examples
    y (batch_size, 5): ground truth labels
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
    pred_labels = np.array(probabilities.numpy() > thr, dtype=float)  # (batch_size, n_classes)
    return pred_labels

def evaluate(model, dataloader, thr, gpu_id=None):
    """
    model: Pytorch model
    X (batch_size, 1000, 3) : batch of examples
    y (batch_size,5): ground truth labels_train
    """
    model.eval()  # set dropout and batch normalization layers to evaluation mode
    with torch.no_grad():
        matrix = np.zeros((5, 5))
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

def compute_loss(model, dataloader, criterion, gpu_id=None):
    #compute the validation loss at the end of each epoch
    model.eval()
    with torch.no_grad():
        val_losses = []
        for i, (x_batch, y_batch) in enumerate(dataloader):
            print('eval {} of {}'.format(i + 1, len(dataloader)), end='\r')
            x_batch, y_batch = x_batch.to(gpu_id), y_batch.to(gpu_id)
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            val_losses.append(loss.item())
            #delete unnecessary variables due to memory issues
            del x_batch
            del y_batch
            torch.cuda.empty_cache()
        model.train()
        return statistics.mean(val_losses)

def threshold_optimization(model, dataloader, gpu_id=None):
    """
    Make labels_train predictions for "X" (batch_size, 1000, 3)
    """
    model.eval()
    with torch.no_grad():
        threshold_opt = np.zeros(5)
        for _, (X, Y) in enumerate(dataloader):
            X, Y = X.to(gpu_id), Y.to(gpu_id)

            Y = np.array(Y.cpu())
            #print(Y)

            logits_ = model(X)  # (batch_size, n_classes)
            probabilities = torch.sigmoid(logits_).cpu()

            # find the optimal threshold with ROC curve for each disease

            for dis in range(0, 5):
                # print(probabilities[:, dis])
                # print(Y[:, dis])
                fpr, tpr, thresholds = roc_curve(Y[:, dis], probabilities[:, dis])
                #print('opt')
                #print(thresholds)
                # weighted mean of sensitivity and specificity (using the number of samples)
                gmean = (9857/17111)*tpr+(7254/17111)*(1-fpr)

                #remove first element
                thresholds = thresholds[1:]
                gmean = gmean[1:]

                # optimal threshold
                index = np.argmax(gmean)
                threshold_opt[dis] = round(thresholds[index], ndigits=2)

    return threshold_opt    
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', default=None,
                        help="Path to the dataset.")
    parser.add_argument('-epochs', default=100, type=int,
                        help="""Number of epochs to train the model.""")
    parser.add_argument('-batch_size', default=4, type=int,
                        help="Size of training batch.")
    parser.add_argument('-learning_rate', type=float, default=0.01)
    parser.add_argument('-l2_decay', type=float, default=0)
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('-gpu_id', type=int, default=None)
    parser.add_argument('-path_save_model', default=None,
                        help='Path to save the model')
    opt = parser.parse_args()

    configure_seed(seed=42)
    configure_device(opt.gpu_id)

    print("Loading data...") ## input manual nexamples train, dev e test
    train_dataset = ECGImageDataset(opt.data, samples, 'train')
    dev_dataset = ECGImageDataset(opt.data, samples, 'dev')
    test_dataset = ECGImageDataset(opt.data, samples, 'test')

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=opt.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)


    n_classes = 5  # 5 diseases + normal

    # initialize the model
    model = AlexNet(n_classes)
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
    class_weights=torch.tensor(class_weight,dtype=torch.float)  
    class_weights = class_weights.to(opt.gpu_id)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights) #https://learnopencv.com/multi-label-image-classification-with-pytorch-image-tagging/
    # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html

    # training loop
    epochs = torch.arange(1, opt.epochs + 1)
    train_mean_losses = []
    valid_mean_losses = []
    valid_specificity = []
    valid_sensitivity = []
    train_losses = []
    last_valid_loss = 100000
    patience_count = 0
    epochs_plot = []    
    for ii in epochs:
        print('Training epoch {}'.format(ii))
        epochs_plot.append(ii)        
        for i, (X_batch, y_batch) in enumerate(train_dataloader):
            print('{} of {}'.format(i + 1, len(train_dataloader)), end='\r', flush=True)
            loss = train_batch(
                X_batch, y_batch, model, optimizer, criterion, gpu_id=opt.gpu_id)
            del X_batch
            del y_batch
            torch.cuda.empty_cache()
            train_losses.append(loss)

        mean_loss = torch.tensor(train_losses).mean().item()
        print('Training loss: %.4f' % (mean_loss))

        train_mean_losses.append(mean_loss)
        #sensitivity, specificity = evaluate(model, dev_dataloader, 'dev', gpu_id=opt.gpu_id)
        val_loss = compute_loss(model, dev_dataloader, criterion, gpu_id=opt.gpu_id)
        valid_mean_losses.append(val_loss)
        #valid_sensitivity.append(sensitivity)
        #valid_specificity.append(specificity)
        #print('Valid specificity: %.4f' % (valid_specificity[-1]))
        #print('Valid sensitivity: %.4f' % (valid_sensitivity[-1]))
        
        if val_loss<last_valid_loss:
            #https://pytorch.org/tutorials/beginner/saving_loading_models.html (save the best model based on the validation loss)
            torch.save(model.state_dict(), os.path.join(opt.path_save_model, model.__class__.__name__ + '_ep_'+ str(ii.item())))
            last_valid_loss = val_loss
            patience_count = 0
        else:
        	patience_count += 1            

        if patience_count==20:
        	plot_losses(valid_mean_losses, train_mean_losses, ylabel='Loss', name='training-validation-loss-{}-{}'.format(opt.learning_rate, opt.optimizer))
        	break            
            
            
    # Results on test set:
    thr = threshold_optimization(model, test_dataloader, gpu_id=opt.gpu_id)
    matrix = evaluate(model, test_dataloader, thr, gpu_id=opt.gpu_id)
    print(matrix)
    metrics = compute_metrics(matrix, class_names=class_names, save_as=f'results/{model.__class__.__name__}')
    print(metrics)

    # plot
    plot_losses(valid_mean_losses, train_mean_losses, ylabel='Loss', name=f'results/figures/{model.__class__.__name__}_loss')

if __name__ == '__main__':
    main()
