import torch
import numpy as np

from torch.utils.data import DataLoader
import gru as gru
import AlexNet as alexnet
import late_fusion as late
import early_fusion as early
import joint_fusion as joint

from count_parameters import count_parameters
from config import samples

type = ['late', 'early', 'joint'][0]

if type == "joint":
    path_weights = "save_models/joint_model_2023-02-03_06-54-57_lr0.01_optadam_dr0.3_eps200_hs256_bs1024_l20.0001"
    thresholds = [0.4761, 0.656, 0.7849, 0.8079]
    batch_size = 1024
    hidden_size = 256
    dropout = 0

elif type == "early":
    path_weights = "save_models/early_model_2023-01-26_01-04-33_lr0.001_optadam_dr0.0_eps200_hs256_bs128_l20"
    thresholds = [0.5659, 0.3952, 0.6742, 0.769]

    batch_size = 128
    hidden_size = 256
    dropout = 0

elif type == "late":
    path_weights = "save_models/late_model_2023-01-22_04-30-10_lr0.1_optadam_dr0.3_eps200_hs512_bs512_l20"
    thresholds = [0.3931, 0.645, 0.6972, 0.8535]

    batch_size = 512
    hidden_size = 512
    dropout = 0

sig_path = "save_models/grubi_dropout05_lr0005_model5"
img_path = "save_models/alexnet"

gpu_id = 0

sig_data = "Dataset/data_for_rnn/"
sig_model = gru.RNN(3, 128, 2, 4, 0.5, gpu_id=gpu_id,
                    bidirectional=True).to(gpu_id)

img_data = "Dataset/Images/"
img_model = alexnet.AlexNet(4).to(gpu_id)

if type == 'late':
    sig_model.load_state_dict(torch.load(sig_path, map_location=torch.device(gpu_id)))
    img_model.load_state_dict(torch.load(img_path, map_location=torch.device(gpu_id)))
    sig_model.eval()
    img_model.eval()

    test_dataset = late.LateFusionDataset(sig_data, img_data, sig_model, img_model, 'gru', 'alexnet',
                                          samples, gpu_id, batch_size, part='test')

    model = late.LateFusionNet(4, 8, hidden_size, dropout).to(gpu_id)

elif type == 'early':
    sig_model.load_state_dict(torch.load(sig_path, map_location=torch.device(gpu_id)))
    sig_model.requires_grad_(False)
    img_model.load_state_dict(torch.load(img_path, map_location=torch.device(gpu_id)))
    img_model.requires_grad_(False)
    sig_model.eval()
    img_model.eval()

    img_hook = 'conv2d_5'
    sig_hook = 'rnn'
    img_model.conv2d_5.register_forward_hook(early.get_activation(img_hook))
    sig_model.rnn.register_forward_hook(early.get_activation(sig_hook))

    test_dataset = early.FusionDataset(sig_data, img_data, samples, part='test')

    model = early.EarlyFusionNet(4, 256, 4096, hidden_size, dropout,
                                 sig_model, img_model, sig_hook, img_hook).to(gpu_id)

else:  # joint fusion
    sig_model.load_state_dict(torch.load(sig_path, map_location=torch.device(gpu_id)))
    img_model.load_state_dict(torch.load(img_path, map_location=torch.device(gpu_id)))
    sig_model.fc = joint.Identity()
    img_model.linear_3 = joint.Identity()

    test_dataset = early.FusionDataset(sig_data, img_data, samples, part='test')

    model = joint.JointFusionNet(4, 256, 2048, hidden_size, dropout, sig_model, img_model).to(gpu_id)

test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model.load_state_dict(torch.load(path_weights, map_location=torch.device(gpu_id)))
model = model.to(gpu_id)
model.eval()

# evaluate the performance of the model
if type == 'late':
    matrix = gru.evaluate(model, test_dataloader, thresholds, gpu_id=gpu_id)
    aurocs = gru.auroc(model, test_dataloader, gpu_id=gpu_id)
else:
    matrix = early.fusion_evaluate(model, test_dataloader, thresholds, gpu_id=gpu_id)
    aurocs = early.fusion_auroc(model, test_dataloader, gpu_id=gpu_id)

count_parameters(model)

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

# Print results
print("Final Test Results:")
for i, cls in enumerate(classes):
    print(f"{cls}: sensitivity - {sensitivities[i]:.2f}; specificity - {specificities[i]:.2f}")
print(f"mean: sensitivity - {mean_sensi:.2f}; specificity - {mean_spec:.2f}")

# Save to file
with open(f'results/model/{str(model.__class__.__name__)}.txt', 'w') as f:
    f.write("Final Test Results:\n")
    for i, cls in enumerate(classes):
        f.write(f"{cls}: sensitivity - {sensitivities[i]:.2f}; specificity - {specificities[i]:.2f}\n")
    f.write(f"mean: sensitivity - {mean_sensi:.2f}; specificity - {mean_spec:.2f}\n")
