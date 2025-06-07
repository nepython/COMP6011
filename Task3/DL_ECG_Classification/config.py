import numpy as np

# Train, Dev, Test samples count
samples = [34530, 4316, 4317]

class_names = ['AFIB', 'AFLT', '1dAVb', 'RBBB', 'LBBB']
classes_n = [5300, 8310, 3717, 6586, 1775] # frequency

# Class weight for correcting imbalance
class_weight = [samples[0]/cn for cn in classes_n]

test_samples = [0, 0, 6]