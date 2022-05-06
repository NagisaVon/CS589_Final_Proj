# Evaluate The Hand-Written Digits Recognition Dataset
# 8*8 numerical attributes, categorical output(0-9),
# 1797 instances 
# Algorithm: KNN/Random Forest/NN

from sklearn import datasets
from evaluation import print_report
from load_data import *
import numpy as np
import random_forest as rf
import decision_tree as dt
import knn as knn
from matplotlib import pyplot as plt


digits = datasets.load_digits()
digits_data = np.vstack((digits['data'].T, digits['target'])).T
digits_attr = digits['feature_names'] + ['class']
digits_attr_type = ["numerical"] * 64 + ["class"]
digits_attr_options = get_possible_options(digits_data, digits_attr_type)
digits_attr_dict = build_attribute_dict(digits_attr, digits_attr_type)
digits_class_col = -1


def digits_dataforharry():
        return digits_attr_dict, digits_data



def tune_n_tree(list_of_n):
    reports = []
    for n in list_of_n:
        rp = rf.dispatch_k_fold(digits_data, 
            digits_attr, 
            digits_attr_type, 
            digits_attr_options, 
            digits_class_col, 
            minimal_size_for_split=0.,
            minimal_gain=0.,
            maximal_depth=10000,
            algo="entropy",  
            random_state=42, 
            k_fold=10,
            n_trees=n,
            binary_class=False, 
            bootstrap_percentage=0.9,
            debug=True
        )
        print("n = {}".format(n))
        print_report(rp)
        reports.append(rp)
        
    # plot 
    plt.plot(list_of_n, [r[0] for r in reports], label="accuracy")
    plt.plot(list_of_n, [r[1] for r in reports], label="precision")
    plt.plot(list_of_n, [r[2] for r in reports], label="recall")
    plt.plot(list_of_n, [r[3] for r in reports], label="f1")
    plt.xlabel("number of trees")   
    plt.legend()
    plt.title("Digits dataset, tune n_tree")
    plt.savefig("output_fig/digits_tune_n_tree.png")


def tune_max_depth(list_of_max_depth):
    reports = []
    for n in list_of_max_depth:
        rp = rf.dispatch_k_fold(digits_data, 
            digits_attr, 
            digits_attr_type, 
            digits_attr_options, 
            digits_class_col, 
            minimal_size_for_split=0.,
            minimal_gain=0.,
            maximal_depth=n,
            algo="entropy",  
            random_state=42, 
            k_fold=10,
            n_trees=30,
            binary_class=False, 
            bootstrap_percentage=0.9,
            debug=True
        )
        print("max_depth = {}".format(n))
        print_report(rp)
        reports.append(rp)
    # plot 
    plt.plot(list_of_max_depth, [r[0] for r in reports], label="accuracy")
    plt.plot(list_of_max_depth, [r[1] for r in reports], label="precision")
    plt.plot(list_of_max_depth, [r[2] for r in reports], label="recall")
    plt.plot(list_of_max_depth, [r[3] for r in reports], label="f1")
    plt.xlabel("maximal depth")   
    plt.legend()
    plt.title("Digits dataset, tune maximal_depth")
    plt.savefig("output_fig/digits_tune_max_depth.png")


def tune_bootstrap_percentage(list_of_bootstrap_percentage):
    reports = []
    for n in list_of_bootstrap_percentage:
        rp = rf.dispatch_k_fold(digits_data, 
            digits_attr, 
            digits_attr_type, 
            digits_attr_options, 
            digits_class_col, 
            minimal_size_for_split=0.,
            minimal_gain=0.,
            maximal_depth=11,
            algo="entropy",  
            random_state=42, 
            k_fold=10,
            n_trees=30,
            binary_class=False, 
            bootstrap_percentage=n,
            debug=True
        )
        print("list_of_bootstrap_percentage = {}".format(n))
        print_report(rp)
        reports.append(rp)
    # plot 
    plt.plot(list_of_bootstrap_percentage, [r[0] for r in reports], label="accuracy")
    plt.plot(list_of_bootstrap_percentage, [r[1] for r in reports], label="precision")
    plt.plot(list_of_bootstrap_percentage, [r[2] for r in reports], label="recall")
    plt.plot(list_of_bootstrap_percentage, [r[3] for r in reports], label="f1")
    plt.xlabel("bootstrap percentage")   
    plt.legend()
    plt.title("Digits dataset, tune bootstrap_percentage")
    plt.savefig("output_fig/digits_tune_bootstrap_percentage.png")


def tune_algo():
    reports = []
    rp = rf.dispatch_k_fold(digits_data, 
            digits_attr, 
            digits_attr_type, 
            digits_attr_options, 
            digits_class_col, 
            minimal_size_for_split=0.,
            minimal_gain=0.,
            maximal_depth=11,
            algo="entropy",  
            random_state=42, 
            k_fold=10,
            n_trees=30,
            binary_class=False, 
            bootstrap_percentage=0.8
        )
    print_report(rp)
    rp = rf.dispatch_k_fold(digits_data, 
            digits_attr, 
            digits_attr_type, 
            digits_attr_options, 
            digits_class_col, 
            minimal_size_for_split=0.,
            minimal_gain=0.,
            maximal_depth=11,
            algo="gini",  
            random_state=42, 
            k_fold=10,
            n_trees=20,
            binary_class=False, 
            bootstrap_percentage=0.8
        )
    print_report(rp)


def digits_rf_final():
    rp = rf.dispatch_k_fold(digits_data, 
            digits_attr, 
            digits_attr_type, 
            digits_attr_options, 
            digits_class_col, 
            minimal_size_for_split=0.,
            minimal_gain=0.,
            maximal_depth=11,
            algo="entropy",  
            random_state=42, 
            k_fold=10,
            n_trees=30,
            binary_class=False, 
            bootstrap_percentage=0.8,
        )
    print("digits random forest final report")
    print_report(rp)


def tune_knn_k(list_of_k):
    reports = []
    for n in list_of_k:
        rp = knn.dispatch_k_fold(digits_data, digits_attr, digits_attr_type, digits_attr_options, digits_class_col, n, k_fold=10, binary_class=False, debug=True)
        print("k = {}".format(n))
        print_report(rp)
        reports.append(rp)
    plt.plot(list_of_k, [r[0] for r in reports], label="accuracy")
    plt.plot(list_of_k, [r[1] for r in reports], label="precision")
    plt.plot(list_of_k, [r[2] for r in reports], label="recall")
    plt.plot(list_of_k, [r[3] for r in reports], label="f1")
    plt.xticks(list_of_k)
    plt.xlabel("k value for kNN")   
    plt.legend()
    plt.title("Digits dataset, tune knn_k")
    plt.savefig("output_fig/digits_tune_knn_k.png")
    print_report(rp)


if __name__ == "__main__":
    # tune random forest
    list_of_n = [1, 5, 10, 20, 30, 40, 50] 
    # tune_n_tree(list_of_n)  # picked n = 20
    # dt.dispatch_decision_tree(digits_data, digits_attr, digits_attr_type, digits_attr_options, digits_class_col, algo="entropy", minimal_size_for_split=0., minimal_gain=0., maximal_depth=10000,random_state=42, printTree=True)
    list_of_max_depth = [8, 9, 10, 11, 12, 13] # picked max_depth = 12
    # tune_max_depth(list_of_max_depth)
    list_of_bootstrap_percentage = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95] # picked bootstrap_percentage = 0.8
    # tune_bootstrap_percentage(list_of_bootstrap_percentage)
    # tune_algo() # picked entropy
    # entropy: accuracy: 0.945  precision: 0.950  recall: 0.945  f1: 0.944
    # gini: accuracy: 0.924  precision: 0.931  recall: 0.924  f1: 0.923
    digits_rf_final()  # accuracy: 0.945  precision: 0.950  recall: 0.945  f1: 0.944
    # tune_knn_k(list(range(1, 20))) # picked k = 3 accuracy: 0.978  precision: 0.980  recall: 0.978  f1: 0.978
