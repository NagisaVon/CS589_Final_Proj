from collections import Counter
import math
import numpy as np
import random
from evaluation import *

def create_bootstrap_replace(data, n_samples):
    # random.sample does not works for np.array
    data = data.tolist()
    bootstrap_data = random.sample(data, n_samples)
    n_replace = len(data) - n_samples
    replace_data = random.sample(data, n_replace)
    return np.array(bootstrap_data + replace_data)


# return prediction 
def brute_k_nearest_neighbor(k, data, point, dist_func): 
    # take away labels for distance calculation
    data_no_labels = [row[:-1] for row in data]
    # calculate distances between point and all data points
    dist = [dist_func(point, data_no_labels[i]) for i in range(len(data))]
    # sort distances and store the indices of the k closest points
    ind =  np.argsort(dist)[:k]
    # labels of the k closest points
    labels = [data[i][-1] for i in ind]
    # return the prediction 
    return Counter(labels).most_common()[0][0]


# return the a report [accuracy, precision, recall, f1] 
def eval_k_nearest_neighbor(k, train_set, test_set, tag_col, class_list, binary_class):
    true_tags = test_set.T[tag_col]
    pred_tags = []
    for test_row in test_set:
        # math.dist is euclidean distance
        prediction = brute_k_nearest_neighbor(k, train_set, test_row[:-1], math.dist)
        pred_tags.append(prediction)
    confusion_matrix = build_confusion_matrix(true_tags, pred_tags, class_list)
    report = build_report(confusion_matrix, len(true_tags), class_list, binary_class)
    return report


def dispatch_k_fold(data, attr, attr_type, attr_opt, tag_col, kNN_k, k_fold=10, binary_class=False, debug=False):
    if debug:
        print("dispatching_kNN_k_fold, setting: kNN_k: {}, k_fold: {}".format(kNN_k, k_fold))

    # k_fold_size = int(len(data) / k_fold)
    data_by_class = [] 
    for i in range(len(attr_opt[tag_col])):
        tag = attr_opt[tag_col][i]
        data_by_class.append(data[data.T[tag_col] == tag])
        data_by_class[i] = np.array_split(data_by_class[i], k_fold)
    
    # [[accuracy, precision, recall, f1], ...]
    k_fold_eval = []
    for i in range(k_fold):
        train = np.array([]).reshape(0, len(data.T))
        test = np.array([]).reshape(0, len(data.T))
        # build train and test from different classes
        for j in range(len(attr_opt[tag_col])):
            # test set is the the i-th fold
            test = np.vstack([test, data_by_class[j][i]])
            # train set is all data except the i-th fold
            # this is so stupid, but it works
            before_i = np.array([]).reshape(0, len(data.T))
            after_i =  np.array([]).reshape(0, len(data.T))
            for k in range(i):
                before_i = np.vstack([before_i, data_by_class[j][k]])
            for k in range(i+1, k_fold):
                after_i = np.vstack([after_i, data_by_class[j][k]])
            if (len(before_i) != 0):
                train = np.vstack([train ,before_i])
            if (len(after_i) != 0):
                train = np.vstack([train ,after_i])

        # print(len(train)+len(test)==len(data))
        eval_result = eval_k_nearest_neighbor(kNN_k, train, test, tag_col, attr_opt[tag_col], binary_class)
        k_fold_eval.append(eval_result)
        if debug:
            print("finished {}-th fold".format(i))
            print_report(eval_result)
    
    k_fold_eval = np.mean(k_fold_eval, axis=0)
    return k_fold_eval 