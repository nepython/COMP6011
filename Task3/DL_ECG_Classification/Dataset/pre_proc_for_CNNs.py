import os
import numpy as np
import tifffile
import cv2
import pickle
from scipy.signal import butter, sosfilt
from pyts.image import GramianAngularField
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm
from joblib import Parallel, delayed

# Pre-compute static stuff
band_pass_sos = butter(2, [1, 45], 'bandpass', fs=100, output='sos')
gaf_transform = GramianAngularField(image_size=1000, method='summation')
quantiles = np.linspace(0, 1, 11)[1:]  # 10 bins

def ecgnorm_batch(X):
    # X shape: (T, n_leads)
    mins = X.min(axis=0, keepdims=True)
    maxs = X.max(axis=0, keepdims=True)
    return (X - mins) / (maxs - mins + 1e-8)

def recurrence_plot(ecg, eps=0.1, steps=10):
    d = pairwise_distances(ecg[:, None]) / eps
    d[d > steps] = steps
    return d / 5. - 1

def get_mtf(ecg, bins=quantiles):
    # vectorize quantile assignment
    q = np.digitize(ecg, bins)
    # transition matrix
    K = len(bins)
    M = np.zeros((K, K), dtype=float)
    for a, b in zip(q[:-1], q[1:]):
        M[a, b] += 1
    M /= (M.sum(axis=1, keepdims=True) + 1e-8)
    # map back
    return M[q[:, None], q[None, :]] / 5. - 1

def ecg_to_3images(ecg_lead):
    # assumes ecg_lead is already filtered & normed
    gaf = (gaf_transform.transform(ecg_lead[None, :]) + 1) / 2
    rp  = recurrence_plot(ecg_lead)
    mtf = get_mtf(ecg_lead)
    return np.stack([gaf[0], mtf, (rp + 1)/2], axis=0)  # shape (12,1000,1000)

def resize_stack(stack, size=(256,256)):
    # stack shape: (C, H, W)
    return np.array([cv2.resize(img, size) for img in stack])

def process_example(X_i):
    # 1) batch filter & normalize
    X_filt = sosfilt(band_pass_sos, X_i, axis=0)
    X_norm = ecgnorm_batch(X_filt)

    # 2) parallel lead transforms
    lead_images = Parallel(n_jobs=-1)(
        delayed(ecg_to_3images)(X_norm[:, lead])
        for lead in range(12)
    )
    # concatenate: (12,3,1000,1000) → (36,1000,1000)
    raw_stack = np.concatenate(lead_images, axis=0)

    # 3) resize once
    return resize_stack(raw_stack)

def X_for_CNNs(path, partition='train', save_dir=None):
    with open(f"{path}/X_{partition}_processed.pickle", "rb") as f:
        X = np.array(pickle.load(f))  # shape (N,1000,12)

    out_dir = os.path.join(save_dir, f'X_cnn_{partition}')
    os.makedirs(out_dir, exist_ok=True)

    for i in tqdm(range(X.shape[0]), desc='2D processing'):
        stacked = process_example(X[i])
        # scale & save
        img = (stacked * 255).astype('uint8')
        tifffile.imwrite(f"{out_dir}/{i}.tif", img)
