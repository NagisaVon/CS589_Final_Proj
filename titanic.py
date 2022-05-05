# The Titanic Dataset
# 8 attributes including class
# 887 instances 
# Algorithm: Random Forest/NN
# the class attribute = [0, 1]


from evaluation import print_report
from load_data import *
import numpy as np
import random_forest as rf
import decision_tree as dt
import knn as knn
from matplotlib import pyplot as plt


# load data and the metadata
titanic_attr_type = ["class","categorical","categorical","categorical",
        "numerical","numerical","numerical","numerical"]
titanic_data, titanic_attr = load_data_category_string("datasets/titanic.csv", titanic_attr_type, csv_delimiter=',', nameToPrefix=True)
# changed name to a categorical attribute
titanic_attr_dict = build_attribute_dict(titanic_attr, titanic_attr_type)
titanic_attr_options = get_possible_options(titanic_data, titanic_attr_type)
titanic_class_col = 0

def titanic_dataforharry():
        return titanic_attr_dict, titanic_data

def tune_n_tree(list_of_n):
    reports = []
    for n in list_of_n:
        rp = rf.dispatch_k_fold(titanic_data, 
            titanic_attr, 
            titanic_attr_type, 
            titanic_attr_options, 
            titanic_class_col, 
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
    plt.title("titanic dataset, tune n_tree")
    plt.savefig("output_fig/titanic_tune_n_tree.png")


if (__name__) == "__main__":
    tune_n_tree([1, 5, 10, 20, 30, 40, 50]) # pick n=30, accuracy: 0.852  precision: 0.818  recall: 0.794  f1: 0.806 


