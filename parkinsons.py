# Evaluate The Oxford Parkinson’s Disease Detection Dataset
# 22 numerical attributes, binary output,
# 195 instances 
# Algorithm: KNN/Random Forest/NN

from load_data import * 
import random_forest as rf
import knn as knn
from matplotlib import pyplot as plt
from evaluation import print_report


# load data and the metadata
parkinsons_data, parkinsons_attr = load_data("datasets/parkinsons.csv", csv_delimiter=',')
parkinsons_attr_type = ["numerical","numerical","numerical","numerical","numerical",
        "numerical","numerical","numerical","numerical","numerical",
        "numerical","numerical","numerical","numerical","numerical",
        "numerical","numerical","numerical","numerical","numerical","numerical","numerical","class"]
parkinsons_attr_dict = build_attribute_dict(parkinsons_attr, parkinsons_attr_type)
parkinsons_attr_options = get_possible_options(parkinsons_data, parkinsons_attr_type)
parkinsons_class_col = -1


def dataforharry():
        return parkinsons_attr_dict, parkinsons_data


def tune_n_tree(list_of_n):
    reports = []
    for n in list_of_n:
        rp = rf.dispatch_k_fold(parkinsons_data, 
            parkinsons_attr, 
            parkinsons_attr_type, 
            parkinsons_attr_options, 
            parkinsons_class_col, 
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
    plt.title("parkinsons dataset, tune n_tree")
    plt.savefig("output_fig/parkinsons_tune_n_tree.png")


def tune_knn_k(list_of_k):
    reports = []
    for n in list_of_k:
        rp = knn.dispatch_k_fold(parkinsons_data, parkinsons_attr, parkinsons_attr_type, parkinsons_attr_options, parkinsons_class_col, n, k_fold=10, binary_class=False, debug=True)
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
    plt.title("parkinsons dataset, tune knn_k")
    plt.savefig("output_fig/parkinsons_tune_knn_k.png")
    print_report(rp)

if (__name__) == "__main__":
    tune_n_tree([1, 5, 10, 20, 30, 40, 50, 60, 70, 90, 120])  # picked n_tree = 90  accuracy: 0.889  precision: 0.875  recall: 1.000  f1: 0.933
    # tune_knn_k(list(range(1, 20)))  # picked k = 14, accuracy: 0.833  precision: 0.912  recall: 0.625  f1: 0.652
