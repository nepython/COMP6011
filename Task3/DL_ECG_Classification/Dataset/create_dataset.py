# Create and save dataset to feed the networks:
# Pre-processing of data (X) to be used in the RNNs and creation images to be used in the 2DCNNs.

from pre_proc_for_rnns import X_for_RNNs, labels
from pre_proc_for_CNNs import X_for_CNNs
# import create_images
import argparse
import os

# path to the "Processed" data folder https://drive.google.com/drive/folders/1Nas7Gqcj-H28Raui_6z06kpWDsM78OBV
parser = argparse.ArgumentParser()
parser.add_argument('-data', default='Processed', help="Path to the dataset.")
parser.add_argument('-save_dir', default='Processed/model_specific', help="Directory to save.")
parser.add_argument('-only_test', default='false', help="Set to true for processing only test set.")
opt = parser.parse_args()

processed_directory = opt.data

# path where the data to feed the models will be stored
path_to_save = opt.save_dir
os.makedirs(path_to_save, exist_ok=True)

print(processed_directory, path_to_save)

# save X_rnn_train
if opt.only_test != 'true':
    X_for_RNNs(processed_directory, 'train', save_dir=path_to_save)
    # save X_rnn_dev
    X_for_RNNs(processed_directory, 'dev', save_dir=path_to_save)
# save X_rnn_test
X_for_RNNs(processed_directory, 'test', save_dir=path_to_save)

if opt.only_test != 'true':
    # save X_cnn_train
    X_for_CNNs(processed_directory, 'train', save_dir=path_to_save)
    # save X_cnn_dev
    X_for_CNNs(processed_directory, 'dev', save_dir=path_to_save)
# save X_cnn_test
X_for_CNNs(processed_directory, 'test', save_dir=path_to_save)

if opt.only_test != 'true':
    # save labels_train
    labels(processed_directory, 'train', save_dir=path_to_save)
    # save labels_dev
    labels(processed_directory, 'dev', save_dir=path_to_save)
# save labels_test
labels(processed_directory, 'test', save_dir=path_to_save)



