import torch
import numpy as np
from prettytable import PrettyTable

from lstm import LSTM
from gru import GRU
from rnn import RNN

from cnn import simplecnn
from AlexNet import AlexNet
from resnet import ResNet50
from vggnet import VGG16

from early_fusion import EarlyFusionNet
from late_fusion import LateFusionNet
from joint_fusion import JointFusionNet

# create a table with the model's parameters
# code from the comments https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
def count_parameters(model, model_name):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    # print(table)
    print(model_name, f"Total Trainable Params: {total_params}")
    return total_params


gpu_id = None
device = 'cpu'

# RNN
model_rnn = RNN(input_size=3, hidden_size=256, num_layers=2, n_classes=5, dropout_rate=0.3, gpu_id=gpu_id, bidirectional=False)

# LSTM
model_lstm = LSTM(input_size=3, hidden_size=256, num_layers=2, n_classes=5, dropout_rate=0.3, gpu_id=gpu_id, bidirectional=False)

# GRU
model_gru = GRU(3, hidden_size=256, num_layers=2, n_classes=5, dropout_rate=0.3, gpu_id=gpu_id,
                bidirectional=False)

# CNN
model_cnn = simplecnn(5)
model_alexnet = AlexNet(5)
model_vggnet = VGG16(5)
model_resnet = ResNet50(5)

# Fusion
# model_earlyfusion = EarlyFusionNet(5)
# model_latefusion = LateFusionNet(5)
# model_jointfusion = JointFusionNet(5)

count_parameters(model_rnn, "RNN")
count_parameters(model_lstm, "LSTM")
count_parameters(model_gru, 'GRU')

count_parameters(model_cnn, 'CNN')
count_parameters(model_alexnet, 'AlexNet')
count_parameters(model_vggnet, 'VGG16')
count_parameters(model_resnet, 'ResNet50')

# count_parameters(model_earlyfusion, 'EarlyFusionNet')
# count_parameters(model_latefusion, 'LateFusionNet')
# count_parameters(model_jointfusion, 'JointFusionNet')
