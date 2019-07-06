from __future__ import division
import numpy as np
from sklearn import metrics
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve

cell_lines = ['29']
#, ]
#'', '']

data_path = '/home/xdjf/fengyutian/workspace/hg19/data/split/'

result_path = '/home/xdjf/fengyutian/workspace/TimHex_hg19/savePredict_9V1/'


def compute_AUPR(y, y_score):
    # print 'Computing Precision-Recall curve...'
    precision, recall, _ = precision_recall_curve(y, y_score)
    #average_precision = average_precision_score(y, y_score)
    return auc(recall, precision)

def compute_AUROC(y, y_score):
    return roc_auc_score(y, y_score)

def max_index(l):
	return [1+l.index(max(l)), max(l)]

def matrix_round(M):
	N = np.zeros((len(M)))
	for index in range(len(M)):
		N[index] =round(M[index])
	return N


for cell in cell_lines:
	precisions = []
	recalls = []
	accuracys = []
	error_rates = []
	F1s = []
	aurocs = []
	auprs = []
	TPs = []
	FPs = []
	TNs = []
	FNs = []

	y_true = np.load(data_path + cell + '_labels_test.npy')
	y_true = y_true.reshape([-1])
	#print(y_true.sum())
	#assert(1==0)
	

	for i in range(40):
		y_pred = np.load(result_path + cell  + str(i+1) + '.npy')
		#print(y_pred)
		y_pred = y_pred.reshape([-1])
		N = matrix_round(y_pred)
		#print(y_pred.shape)

		#print(y_true)
		#print(N)
		#assert(1==0)


		TP = np.sum(np.logical_and(np.equal(y_true,1),np.equal(N,1)))

		FP = np.sum(np.logical_and(np.equal(y_true,0),np.equal(N,1)))
		TN = np.sum(np.logical_and(np.equal(y_true,0),np.equal(N,0)))
		FN = np.sum(np.logical_and(np.equal(y_true,1),np.equal(N,0)))

		#print(TP)
		#assert(1==0)

		precision = TP / (TP + FP)
		recall = TP / (TP + FN)
		accuracy = (TP + TN) / (TP + FP + TN + FN)
		error_rate =  (FN + FP) / (TP + FP + TN + FN)
		F1 = 2*precision*recall/(precision+recall)

		#classify_report = metrics.classification_report(y_true, y_pred)
		#confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
		#overall_accuracy = metrics.accuracy_score(y_true, y_pred)
		#acc_for_each_class = metrics.precision_score(y_true, y_pred, average=None)
		#average_accuracy = np.mean(acc_for_each_class)
		#score = metrics.accuracy_score(y_true, y_pred)

		TPs.append(TP)
		FPs.append(FP)
		TNs.append(TN)
		FNs.append(FN)
		precisions.append(precision)
		recalls.append(recall)
		accuracys.append(accuracy)
		error_rates.append(error_rate)
		F1s.append(F1)
		aurocs.append(compute_AUROC(y_true, y_pred))
		auprs.append(compute_AUPR(y_true, y_pred))

	print("%s %d" %(cell, y_true.shape[0]))
	print("precision max:")
	print(max_index(precisions))
	index = max_index(precisions)[0]-1
	print("TP:%d FP:%d TN:%d FN:%d"%(TPs[index], FPs[index], TNs[index], FNs[index]))
	print("recall max:")
	print(max_index(recalls))
	print("accuracy max:")
	print(max_index(accuracys))
	print("F1 max:")
	print(max_index(F1s))
	print("aurocs max:")
	print(max_index(aurocs))
	print("auprs:")
	print(max_index(auprs))


