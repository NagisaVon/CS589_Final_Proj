# The Loan Eligibility Prediction Dataset
# 8 categorical attributes, 4 numerical attributes
# 480 instances 
# Algorithm: Random Forest/NN
# the class attribute = [Y, N]

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


def loan_dataforharry():
        return loan_attr_dict, loan_data


def tune_n_tree(list_of_n):
    reports = []
    for n in list_of_n:
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
            n_trees=n,
            binary_class=True, 
            bootstrap_percentage=0.9,
            debug=True
        )
        print("n = {}".format(n))
        print_report(rp)
        reports.append(rp)
    plt.plot(list_of_n, [r[0] for r in reports], label="accuracy")
    plt.plot(list_of_n, [r[1] for r in reports], label="precision")
    plt.plot(list_of_n, [r[2] for r in reports], label="recall")
    plt.plot(list_of_n, [r[3] for r in reports], label="f1")
    plt.xticks(list_of_n)
    plt.xlabel("Number of trees")   
    plt.legend()
    plt.title("Loan dataset, tune n_tree")
    plt.savefig("output_fig/loan_tune_n_tree.png")


if (__name__) == "__main__":
    tune_n_tree([1, 5, 10, 20, 30, 40, 50])