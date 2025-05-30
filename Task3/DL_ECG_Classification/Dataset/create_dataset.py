# Create and save dataset to feed the networks:
# Pre-processing of data (X) to be used in the RNNs and creation images to be used in the 2DCNNs.

from pre_proc_for_rnns import X_for_RNNs, labels
from pre_proc_for_CNNs import X_for_CNNs
import create_images
import os

# path to the "Processed" data folder https://drive.google.com/drive/folders/1Nas7Gqcj-H28Raui_6z06kpWDsM78OBV
processed_directory = 'Processed' #None

# path where the data to feed the models will be stored
path_to_save = 'Processed/model_specific' #None
os.makedirs(path_to_save, exist_ok=True)

# save X_rnn_train
X_for_RNNs(processed_directory, 'train', save_dir=path_to_save)
# save X_rnn_dev
X_for_RNNs(processed_directory, 'dev', save_dir=path_to_save)
# save X_rnn_test
X_for_RNNs(processed_directory, 'test', save_dir=path_to_save)

# save X_cnn_train
X_for_CNNs(processed_directory, 'train', save_dir=path_to_save)
# save X_cnn_dev
X_for_CNNs(processed_directory, 'dev', save_dir=path_to_save)
# save X_cnn_test
X_for_CNNs(processed_directory, 'test', save_dir=path_to_save)

# save labels_train
labels(processed_directory, 'train', save_dir=path_to_save)
# save labels_dev
labels(processed_directory, 'dev', save_dir=path_to_save)
# save labels_test
labels(processed_directory, 'test', save_dir=path_to_save)



