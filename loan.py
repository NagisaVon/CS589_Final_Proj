# The Loan Eligibility Prediction Dataset
# 8 categorical attributes, 4 numerical attributes
# 480 instances 
# Algorithm: Random Forest/NN
# the class attribute = [Y, N]

from sklearn import datasets
from evaluation import print_report
from load_data import *
import numpy as np
import random_forest as rf
import decision_tree as dt
import knn as knn
from matplotlib import pyplot as plt

# load data and the metadata
loan_attr_type = ["categorical","categorical","categorical","categorical",
        "categorical","numerical","numerical","numerical","numerical",
        "categorical","categorical", "class"]
loan_data, loan_attr = load_data_category_string("datasets/loan.csv", loan_attr_type, csv_delimiter=',', dropID=True)
# first attribute is ID, is dropped
loan_attr_dict = build_attribute_dict(loan_attr, loan_attr_type)
loan_attr_options = get_possible_options(loan_data, loan_attr_type)
loan_class_col = -1


rp = rf.dispatch_k_fold(loan_data, 
            loan_attr, 
            loan_attr_type, 
            loan_attr_options, 
            loan_class_col, 
            minimal_size_for_split=0.,
            minimal_gain=0.,
            maximal_depth=10000,
            algo="entropy",  
            random_state=42, 
            k_fold=10,
            n_trees=10,
            binary_class=True, 
            bootstrap_percentage=0.9,
            debug=True
        )
        
print_report(rp)