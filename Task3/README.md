## ECG multi-label classification
1) Dataset
`FILES`: Preprocessing.ipynb
* We use the aggregate dataset provided by `PhysioNet/Computing in Cardiology Challenge 2021` which is a collection of eight different datasets with varying frequencies, duration and diagnosis. These were standardised by us to 100Hz, 10s and limiting to our conditions of interest: `AFIB`, `AFLT`, `LBBB`, `RBBB`, `1dAVB`, `NORM`.
* Next, we preprocess the signals using BandPass and z-score normalisation.
* This is followed by creating 1d and 2d arrays for RNN and CNN design inputs. Labels are encoded using one-hot encoding.
2) Modelling
`FILES`: models/oned.py, models/twod.py, DL_ECG_Classification/early_fusion.py, DL_ECG_Classification/late_fusion.py, DL_ECG_Classification/joint_fusion.py
3) Training
`FILES`: Train-CC21.ipynb
4) Evaluation
`FILES`: Evaluation.ipynb
5) Interpretation
`FILES`: Interpret.ipynb
