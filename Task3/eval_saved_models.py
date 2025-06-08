import torch
import numpy as np
from torch.utils.data import DataLoader
from utils import Dataset_for_RNN, ECGImageDataset, configure_device, configure_seed, compute_metrics
import argparse
from models.oned import GRU, LSTM, RNN
from models.twod import CNN
from models.twod import VGG16 as VGG
from models.twod import AlexNet
from models.twod import ResNet50 as ResNet
from config import test_samples as samples, class_weight, class_names
from utils import predict, threshold_optimization, auroc, evaluate_with_norm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu_id', type=int, default=None)
    parser.add_argument('-model', type=str, default='gru')
    parser.add_argument('-datatype', type=str, default='rnn')
    parser.add_argument('-path', type=str, default='')
    parser.add_argument('-data', type=str, default='data_for_rnn/')
    parser.add_argument('-predict_only', type=bool, default=False)
    parser.add_argument('-num_layers', type=int, default=2)
    parser.add_argument('-hidden_size', type=int, default=256)
    parser.add_argument('-bidirectional', type=bool, default=False)
    opt = parser.parse_args()

    configure_seed(seed=42)
    configure_device(opt.gpu_id)

    gpu_id = opt.gpu_id
    device = 'cuda'
    if gpu_id is None:
        device = 'cpu'

    input_size = 12
    hidden_size = opt.hidden_size
    num_layers = opt.num_layers
    n_classes = 5

    # choose the model for evaluation on test set
    if opt.model == 'rnn':
        model = RNN(input_size, hidden_size, num_layers, n_classes, dropout_rate=0, gpu_id=opt.gpu_id)
    elif opt.model == 'lstm':
        model = LSTM(input_size, hidden_size, num_layers, n_classes, dropout_rate=0, gpu_id=opt.gpu_id,
                    bidirectional=opt.bidirectional)
    elif opt.model == 'gru':
        model = GRU(input_size, hidden_size, num_layers, n_classes, dropout_rate=0, gpu_id=opt.gpu_id,
                    bidirectional=opt.bidirectional)
    elif opt.model == 'cnn':
        model = CNN(n_classes)
    elif opt.model == 'alexnet':
        model = AlexNet(n_classes)
    elif opt.model == 'vgg':
        model = VGG(n_classes)
    elif opt.model == 'resnet':
        model = ResNet(n_classes)
    model.load_state_dict(torch.load(opt.path))
    model.to(opt.gpu_id)

    # model in the evaluation mode
    model.eval()

    # test dataset
    if opt.datatype == 'rnn':
        test_dataset = Dataset_for_RNN(opt.data, samples, 'test')
    elif opt.datatype == 'cnn':
        test_dataset = ECGImageDataset(opt.data, samples, 'test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # # dev dataset
    # dev_dataset = Dataset_for_RNN(path_to_data, samples, 'dev')
    # dev_dataloader = DataLoader(dev_dataset, batch_size=512, shuffle=False)

    if opt.predict_only:
        print(f'Predicted probabilities')
        preds = np.zeros(shape=(len(test_dataset), 5))
        for i, (x_batch, y_batch) in enumerate(test_dataloader):
            x_batch, y_batch = x_batch.to(gpu_id), y_batch.to(gpu_id)
            y_pred = predict(model, x_batch, None)
            preds[i, :] = y_pred.detach().cpu()
        print(np.around(preds, 2))
        return
            

    # threshold optimization
    thr = threshold_optimization(model, test_dataloader, gpu_id=gpu_id)

    # Results on test set:
    matrix, norm_vec = evaluate_with_norm(model, test_dataloader, thr, gpu_id=opt.gpu_id)
    matrix = np.vstack([matrix, norm_vec])
    # matrix = evaluate(model, test_dataloader, thr, gpu_id=None)
    # aurocs = auroc(model, test_dataloader, gpu_id=gpu_id)

    print(matrix)
    metrics = compute_metrics(matrix, class_names=class_names+['NORM'], save_as=f'test_results/{model.__class__.__name__}')
    print(metrics)

    # Print results
    # print("Final Test Results:")
    # for metric, values in metrics.items():
    #     for i, cls in enumerate(classes):
    #         print(f"{cls}: {metric} - {values[i]:.2f}")
    #     print(f"mean: {metric} - {mean_values[metric]:.2f}")
    # print(f"mean: G-Mean - {mean_values['g_mean']:.2f}")

if __name__ == '__main__':
    main()
