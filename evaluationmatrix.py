import math
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

# I reused the code from my last assignment.

def accuracy(truePosi, trueNega, falsePosi, falseNega): # Count of all four
	return (truePosi+trueNega)/(truePosi+trueNega+falseNega+falsePosi)

def precision(truePosi, trueNega, falsePosi, falseNega):
	if (truePosi+falsePosi) == 0:
		return 0
	preposi = truePosi/(truePosi+falsePosi)
	# prenega = trueNega/(trueNega+falseNega)
	return preposi

def recall(truePosi, trueNega, falsePosi, falseNega):
	if (truePosi+falseNega)== 0:
		return 0
	recposi = truePosi/(truePosi+falseNega)
	# recnega = trueNega/(trueNega+falsePosi)
	return recposi

def fscore(truePosi, trueNega, falsePosi, falseNega, beta: 1):
	pre = precision(truePosi, trueNega, falsePosi, falseNega)
	rec = recall(truePosi, trueNega, falsePosi, falseNega)
	if (pre*(beta**2)+rec) == 0:
		return 0
	f = (1+beta**2)*((pre*rec)/(pre*(beta**2)+rec))
	return f

def evaluate(listsofoutput, positivelabel, beta=1):
    # list is list of [predicted, actual]
    listoftptnfpfn = []
    accuarcylists = []
    precisionlists = []
    recalllists = []
    fscorelists = []
    for output in listsofoutput:
        tp, tn, fp, fn, = 0, 0, 0, 0
        for i in range(len(output)):
            if output[i][0] == positivelabel and output[i][1] == positivelabel:
                tp += 1
            elif output[i][0] != positivelabel and output[i][0] == output[i][1]:
                tn += 1
            elif output[i][0] == positivelabel and output[i][1] != positivelabel:
                fp += 1
            elif output[i][0] != positivelabel and output[i][1] == positivelabel:
                fn += 1
        tptnfpfn = [tp, tn, fp, fn]
        listoftptnfpfn.append(tptnfpfn)
        accuarcylists.append(accuracy(tp, tn, fp, fn))
        precisionlists.append(precision(tp, tn, fp, fn))
        recalllists.append(recall(tp, tn, fp, fn))
        fscorelists.append(fscore(tp, tn, fp, fn, beta))
    return accuarcylists, precisionlists, recalllists, fscorelists, listoftptnfpfn

def meanevaluation(listsofoutput, positivelabel, beta=1):
    accuarcylists, precisionlists, recalllists, fscorelists, notused = evaluate(listsofoutput, positivelabel, beta)
    return sum(accuarcylists)/len(accuarcylists), sum(precisionlists)/len(precisionlists), sum(recalllists)/len(recalllists), sum(fscorelists)/len(fscorelists)

def markdownaprf(acc,pre,rec,fsc,beta,nvalue,title):
    acc, pre, rec, fsc = round(acc,3), round(pre,3), round(rec,3), round(fsc,3)
    display(Markdown(rf"""
	Result/Stat of {nvalue} trees random forest of {title}:
    | **Accuracy** | **Precision** | **Recall** | **F-Score, Beta={beta}** |
    | :---: | :---: | :---: | :---: |
    |{acc} | {pre} | {rec} | {fsc} |
    """))

def markdownmatrix(tptnfpfn,title):
    tp, tn, fp, fn = tptnfpfn[0], tptnfpfn[1], tptnfpfn[2], tptnfpfn[3]
    display(Markdown(rf"""
    Confusion Matrix: {title}
    |  | **Predicted +** | **Predicted-** |
    | :--- | :--- | :--- |
    | **Actual +** | {tp} | {fp} |
    | **Actual -** | {fn} | {tn} |
    """))

def confusionmatrix(truePosi, trueNega, falsePosi, falseNega, title=""):
	fig = plt.figure()
	plt.title(title)
	col_labels = ['Predict:+', 'Predict:-']
	row_labels = ['Real:+', 'Real:-']
	table_vals = [[truePosi, falseNega], [falsePosi, trueNega]]
	the_table = plt.table(cellText=table_vals,
                      colWidths=[0.1] * 3,
                      rowLabels=row_labels,
                      colLabels=col_labels,
                      loc='center')
	the_table.auto_set_font_size(False)
	the_table.set_fontsize(24)
	the_table.scale(4, 4)
	plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
	plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)

	for pos in ['right','top','bottom','left']:
		plt.gca().spines[pos].set_visible(False)

	plt.show()	
	return 