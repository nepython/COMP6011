import torch
import numpy as np
from torch.utils.data import DataLoader
from utils import Dataset_for_RNN, configure_device, configure_seed
import argparse
from lstm import LSTM
from cnn_lstm import CNN1d_LSTM
from cnn_gru import CNN1d_GRU
from gru_with_attention import RNN_att
from gru import RNN, threshold_optimization, auroc, evaluate_with_norm
from config import samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu_id', type=int, default=None)
    parser.add_argument('-model', type=str, default='gru')
    parser.add_argument('-path', type=str, default='')
    parser.add_argument('-data', type=str, default='data_for_rnn/')
    opt = parser.parse_args()

    configure_seed(seed=42)
    configure_device(opt.gpu_id)

    gpu_id = opt.gpu_id
    device = 'cuda'
    if gpu_id is None:
        device = 'cpu'

    path_to_data = opt.data

    # choose the model for evaluation on test set

    if opt.model == 'lstm':
        # LSTM
        model_lstm = LSTM(input_size=3, hidden_size=256, num_layers=2, n_classes=4, dropout_rate=0, gpu_id=gpu_id,
                          bidirectional=False)
        model_lstm.load_state_dict(
            torch.load(opt.path or 'best_trained_rnns/lstm_1669240561.183005model28'))
        model = model_lstm.to(opt.gpu_id)
    elif opt.model == 'gru':
        # GRU
        model_gru = RNN(3, hidden_size=128, num_layers=2, n_classes=4, dropout_rate=0, gpu_id=gpu_id,
                        bidirectional=False)
        model_gru.load_state_dict(
            torch.load(opt.path or 'best_trained_rnns/gru_3layers_dropout0_model8'))
        model = model_gru.to(opt.gpu_id)
    elif opt.model == 'bigru':
        # BiGRU
        model_bigru = RNN(3, hidden_size=128, num_layers=2, n_classes=4, dropout_rate=0.5, gpu_id=gpu_id,
                          bidirectional=True)
        model_bigru.load_state_dict(
            torch.load(opt.path or 'best_trained_rnns/grubi_dropout05_lr0005_model5'))
        model = model_bigru.to(opt.gpu_id)
    elif opt.model == 'bigruattention':
        # BiGRU
        model_bigru_att = RNN_att(3, hidden_size=128, num_layers=2, n_classes=4, dropout_rate=0.5, gpu_id=gpu_id,
                          bidirectional=True)
        model_bigru_att.load_state_dict(
            torch.load(opt.path or 'best_trained_rnns/grubi_attention_model4'))
        model = model_bigru_att.to(opt.gpu_id)
    elif opt.model == 'gruattention':
        # BiGRU
        model_gru_att = RNN_att(3, hidden_size=128, num_layers=3, n_classes=4, dropout_rate=0, gpu_id=gpu_id,
                                  bidirectional=False)
        model_gru_att.load_state_dict(
            torch.load(opt.path or 'best_trained_rnns/gru_attention_model4'))
        model = model_gru_att.to(opt.gpu_id)
    elif opt.model == 'cnn_gru':
        # 1D-CNN + GRU
        model_cnn_gru = CNN1d_GRU(input_size=3, hidden_size=256, n_classes=4, dropout_rate=0.3, gpu_id=gpu_id)
        model_cnn_gru.load_state_dict(
            torch.load(opt.path or 'best_trained_rnns/cnn_gru_1672230038.517979model49'))
        model = model_cnn_gru.to(opt.gpu_id)
    elif opt.model == 'cnn_lstm':
        # 1D-CNN + LSTM
        model_cnn_lstm = CNN1d_LSTM(input_size=3, hidden_size=128, n_classes=4, dropout_rate=0.3, gpu_id=gpu_id)
        model_cnn_lstm.load_state_dict(
            torch.load(opt.path or 'best_trained_rnns/cnnlstm_model120'))
        model = model_cnn_lstm.to(opt.gpu_id)
    elif opt.model == 'bilstm':
        # BiLSTM
        model_bilstm = LSTM(input_size=3, hidden_size=128, num_layers=2, n_classes=4, dropout_rate=0, gpu_id=gpu_id,
                            bidirectional=True)
        model_bilstm.load_state_dict(
            torch.load(opt.path or 'best_trained_rnns/lstmbi_dropout05_model20'))
        model = model_bilstm.to(opt.gpu_id)

    # model in the evaluation mode
    model.eval()

    # test dataset
    test_dataset = Dataset_for_RNN(path_to_data, samples, 'test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # dev dataset
    dev_dataset = Dataset_for_RNN(path_to_data, samples, 'dev')
    dev_dataloader = DataLoader(dev_dataset, batch_size=512, shuffle=False)

    # threshold optimization
    thr = threshold_optimization(model, dev_dataloader)

    # evaluate the performance of the model
    matrix, norm_vec = evaluate_with_norm(model, test_dataloader, thr, gpu_id=gpu_id)
    # matrix = evaluate(model, test_dataloader, thr, gpu_id=None)
    aurocs = auroc(model, test_dataloader, gpu_id=gpu_id)

    classes = ["NORM", "AFIB", "AFLT", "1dAVb", "RBBB"]
    metrics = {
        'sensitivity': [],
        'specificity': [],
        'accuracy': [],
        'precision': [],
        'f1_score': [],
        'auroc': []
    }

    # Compute sensitivity and specificity for each class
    for i in range(len(classes)):
        tp, fp, fn, tn = matrix[i]
        sensi = tp / (tp + fn)
        spec = tn / (tn + fp)
        acc = (tp + tn) / np.sum(matrix[i])
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        auroc_value = aurocs[i].item() if i < len(aurocs) else 0
        metrics['sensitivities'].append(sensi)
        metrics['specificities'].append(spec)
        metrics['accuracies'].append(acc)
        metrics['precisions'].append(prec)
        metrics['f1_scores'].append(f1)
        metrics['auroc_values'].append(auroc_value)

    # Compute mean for each metric
    mean_values = {metric: np.mean(values) for metric, values in metrics.items()}
    mean_values['g_mean'] = np.sqrt(mean_values['sensitivities'] * mean_values['specificities'])

    # Print results
    print("Final Test Results:")
    for metric, values in metrics.items():
        for i, cls in enumerate(classes):
            print(f"{cls}: {metric} - {values[i]:.2f}")
        print(f"mean: {metric} - {mean_values[metric]:.2f}")
    print(f"mean: G-Mean - {mean_values['g_mean']:.2f}")

if __name__ == '__main__':
    main()
