from statistics import mean

def build_confusion_matrix(true_tags, pred_tags, class_list):
    '''
    return a confusion matrix, row for predicted tags, column for true tags (or class)
    '''                                                                                                                                                               
    # init matrix
    confusion_matrix = []
    for i in range(len(class_list)):
        row = [0 for j in range(len(class_list))]
        confusion_matrix.append(row)
    for i in range(len(true_tags)):
        pred_ind = class_list.index(int(pred_tags[i]))
        true_ind = class_list.index(int(true_tags[i]))
        confusion_matrix[pred_ind][true_ind] += 1
    return confusion_matrix


def build_report(confusion_matrix, total_entry, class_list, binary_class):
    '''
    return a report of accuracy, precision, recall, f1
    if binary_class is True, then the report use class 1 for positive, and class 0 for negative
    '''
    acc = sum(confusion_matrix[i][i] for i in range(len(confusion_matrix)))/total_entry
    precisions = []
    recalls = []
    f1s = []
    for i in range(len(class_list)):
        tp = confusion_matrix[i][i]
        all_p = sum(confusion_matrix[i][j] for j in range(len(confusion_matrix)))
        fn = sum(confusion_matrix[j][i] for j in range(len(confusion_matrix)) if j != i)
        if all_p == 0:
            precisions.append(0)
        else:
            precisions.append(tp/all_p)
        if tp + fn == 0:
            recalls.append(0)
        else:
            recalls.append(tp/(tp+fn))
        # divide by zero error
        if (precisions[i]+recalls[i]) == 0:
            f1s.append(0)
        else:
            f1s.append(2*precisions[i]*recalls[i]/(precisions[i]+recalls[i]))
    if binary_class:
        return [acc, precisions[1], recalls[1], f1s[1]]
    precision = mean(precisions)
    recall = mean(recalls)
    f1 = mean(f1s)
    return [acc, precision, recall, f1]


def print_report(report):
    print('accuracy: %.3f  precision: %.3f  recall: %.3f  f1: %.3f' %  (report[0], report[1], report[2], report[3]))


def print_markdown(report):
    print( # in jypyter notebook: display(Markdown(
rf"""
| **Accuracy** | **Precision** | **Recall** | **F-Score** |
| :---: | :---: | :---: | :---: |
| {report[0]} | {report[1]} | {report[2]} | {report[3]} |
""")