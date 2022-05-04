# Evaluate The Hand-Written Digits Recognition Dataset
# 8*8 numerical attributes, categorical output(0-9),
# 1797 instances 
# Algorithm: KNN/Random Forest/NN

from sklearn import datasets
import numpy as np
import matplotlib . pyplot as plt

digits = datasets.load_digits()
digits_data  = digits['data']
digits_class = digits['target']
digits_data_with_class = np.vstack((digits_data.T, digits_class)).T
digits_attr = digits['feature_names']
digits_attr_type = ["numerical"] * 64 + ["class"]

print(digits)