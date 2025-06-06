## Create the dataset to train a CNN (AlexNet) to perform ECG classification 
## Based on the GAF, RP and MTF transformations applied to the 1D ECG

#filter the ecg signal (band pass filter)
#select 3 leads (I, II, V2)
#normalize
#convert to images (9,1000,1000) for each example

#save the images as tiff files  (0.tif to number_of_examples.tif) 
#save the labels as numpy arrays (0.tif to number_of_examples.tif)
import cv2
import tifffile 
import numpy as np
import pickle
import os

from joblib import Parallel, delayed
from multiprocessing import Manager
from pyts.image import GramianAngularField, MarkovTransitionField
from scipy.signal import butter, sosfilt
from sklearn.metrics.pairwise import pairwise_distances
from threading import Thread
from tqdm import tqdm


def X_for_CNNs(path, partition='train', save_dir=None):

    file = str(path) + '/X_' + str(partition) + '_processed.pickle'
    pickle_in = open(file, "rb")
    X = pickle.load(pickle_in)

    #band pass filter
    band_pass_filter = butter(2, [1, 45], 'bandpass', fs=100, output='sos')
    for i in tqdm(range(np.shape(X)[0])):
        X_aux = np.zeros((9,1000,1000))

        X_i = X[i] #(1000,12)
        lead_I = X_i[:,0]
        lead_II = X_i[:,1]
        lead_V2 = X_i[:,7]

        #apply a band pass filter (0.05, 40hz)
        lead_I = sosfilt(band_pass_filter, lead_I)
        lead_II = sosfilt(band_pass_filter, lead_II)
        lead_V2 = sosfilt(band_pass_filter, lead_V2)

        #normalize before transforming into images
        lead_I = ecgnorm(lead_I)
        lead_II = ecgnorm(lead_II)
        lead_V2 = ecgnorm(lead_V2)

        #transform each signal into three images
        lead_I_transf = ecgtoimagetransf(lead_I)
        lead_II_transf = ecgtoimagetransf(lead_II)
        lead_V2_transf = ecgtoimagetransf(lead_V2)

        #save in the final array
        X_aux[0:3] = lead_I_transf
        X_aux[3:6] = lead_II_transf
        X_aux[6:9] = lead_V2_transf

        #resizing the image
        X_aux = resizing(X_aux)

        X_aux = X_aux*255.0
        X_aux = X_aux.astype('uint8')
        tifffile.imwrite(str(save_dir) + '/X_cnn_' + str(partition) + '/' + str(i) + '.tif', X_aux)

# def process_ecg_sample(i, X, band_pass_filter, save_dir, partition, progress_queue):
#     X_aux = np.zeros((9, 1000, 1000))
#     X_i = X[i]  # (1000, 12)
#     lead_I = X_i[:, 0]
#     lead_II = X_i[:, 1]
#     lead_V2 = X_i[:, 7]

#     # Apply a band pass filter (0.05, 40hz)
#     lead_I = sosfilt(band_pass_filter, lead_I)
#     lead_II = sosfilt(band_pass_filter, lead_II)
#     lead_V2 = sosfilt(band_pass_filter, lead_V2)

#     # Normalize before transforming into images
#     lead_I = ecgnorm(lead_I)
#     lead_II = ecgnorm(lead_II)
#     lead_V2 = ecgnorm(lead_V2)

#     # Transform each signal into three images
#     lead_I_transf = ecgtoimagetransf(lead_I)
#     lead_II_transf = ecgtoimagetransf(lead_II)
#     lead_V2_transf = ecgtoimagetransf(lead_V2)

#     # Save in the final array
#     X_aux[0:3] = lead_I_transf
#     X_aux[3:6] = lead_II_transf
#     X_aux[6:9] = lead_V2_transf

#     # Resizing the image
#     X_aux = resizing(X_aux)

#     X_aux = X_aux * 255.0
#     X_aux = X_aux.astype('uint8')
#     tifffile.imwrite(os.path.join(save_dir, f'X_cnn_{partition}', f'{i}.tif'), X_aux)

#     # Update progress bar
#     progress_queue.put(1)

# def X_for_CNNs(path, partition='train', save_dir=None):
#     file = os.path.join(path, f'X_{partition}_processed.pickle')
#     with open(file, "rb") as pickle_in:
#         X = pickle.load(pickle_in)

#     # Create output directory if it doesn't exist
#     output_dir = os.path.join(save_dir, f'X_cnn_{partition}')
#     os.makedirs(output_dir, exist_ok=True)

#     # Band pass filter
#     band_pass_filter = butter(2, [1, 45], 'bandpass', fs=100, output='sos')

#     # Initialize progress bar and manager queue
#     manager = Manager()
#     progress_queue = manager.Queue()
#     total_samples = np.shape(X)[0]
#     progress_bar = tqdm(total=total_samples, desc=f'Processing {partition} data')

#     # Function to update progress bar
#     def update_progress_bar(queue, progress_bar):
#         while True:
#             queue.get()
#             progress_bar.update(1)

#     # Start progress bar updater thread
#     progress_thread = Thread(target=update_progress_bar, args=(progress_queue, progress_bar))
#     progress_thread.start()

#     # Parallel processing of ECG samples
#     Parallel(n_jobs=-1, backend='threading')(
#         delayed(process_ecg_sample)(i, X, band_pass_filter, save_dir, partition, progress_queue)
#         for i in range(total_samples)
#     )

#     # Close progress bar
#     progress_bar.close()
#     progress_thread.join()

def ecgnorm(ecg):
    #output between 0 and 1
    ecg_norm = (ecg -min(ecg)) / max(ecg-min(ecg))
    return ecg_norm

def ecgtoimagetransf(ecg):
    aux_img = np.zeros((3,len(ecg), len(ecg)))

    # Gramian Angular Field
    gaf = GramianAngularField(image_size=len(ecg), method='summation')

    x_gaf = gaf.fit_transform(ecg.reshape(1,-1))
    mtf = get_mtf(ecg)
    rp = recurrence_plot(ecg, steps=10)

    x_gaf = (x_gaf+1)/2
    mtf = mtf+1
    rp = (rp+1)/2

    aux_img[0] = x_gaf
    aux_img[1] = mtf
    aux_img[2] = rp 

    return aux_img

#THE FOLLOWING FUNCTIONS ARE ADAPTED FROM https://github.com/zaamad/ECG-Heartbeat-Classification-Using-Multimodal-Fusion

# recurrence plot
def recurrence_plot(s, eps=None, steps=None):
    result = []
    if eps==None: eps=0.1
    if steps==None: steps=10
    d = pairwise_distances(s[:, None])
    d = d / eps
    d[d > steps] = steps
    return d/5. - 1

# Hearbeat values range from 0 to 1; we will divide into quantiles and see which bin each value belongs to
# and what the probability is to transfer from 1 bin to other
# i.e if 1 value is in bin 1 and the next value is in bin 5, what is the probability of transferring from
# bin 1 to 5
# correlation matrix of probabilities is markov transition field
def get_quantiles(min_value=0, max_val=1, k=10):
    c = (max_val - min_value)/k
    b = min_value + c
    d = []
    for i in range(1, k):
        d.append(b)
        b += c
    d.append(max_val)
    return d

quantiles = get_quantiles()

def value_to_quantile(x):
    for i, k in enumerate(quantiles):
        if x <= k:
            return i
    return 0

def get_mtf(x, size=10):
    q = np.vectorize(value_to_quantile)(x)
    r = np.zeros((q.shape[0], q.shape[0]))
    y = np.zeros((size, size))
    for i in range(x.shape[0] - 1):
        y[q[i], q[i + 1]] += 1
    row_sum = y.sum(axis=1, keepdims=True)
    row_sum[row_sum==0] = 1
    y /= row_sum
    y[np.isnan(y)] = 0
    
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            r[i, j] = y[q[i], q[j]]
    return r / 5. - 1


def resizing(X_aux):
    output = []
    for z in range(0,9):
        aux = X_aux[z]
        aux = cv2.resize(aux,(256,256))
        output.append(aux)
    output = np.asarray(output)
    return output
