import numpy as np

# Train, Dev, Test samples count
# samples = [9672,1210,1226]
samples = [8021, 1003, 1003]

class_names = ['AFIB', 'AFLT', '1dAVb', 'RBBB', 'LBBB']
# classes_n = [37, 47, 428, 1039, 405] # frequency
# classes_n = [2213, 330, 839, 1638, 279] # frequency

classes_n = np.array([1., 1., 1., 1., 1.])

# Class weight for correcting imbalance
# class_weight = [samples[0]/max(cn, 1) for cn in classes_n]
class_weight = classes_n

test_samples = [0, 0, 6]