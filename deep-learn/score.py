import numpy as np
from sklearn import metrics
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import math
import sys
folder = sys.argv[1]


y_test = []
for i in range(485):
    y_test.append(1)
for i in range(485):
    y_test.append(0)
y_test = np.array(y_test)
y_test = to_categorical(y_test)


y_score = []
with open(folder,'r') as f:
    for line in f:
        # line = line.split()
        y_score.append(line)
y = np.array(y_score)
y = y.astype(np.float)

print('the Positive probability ' + str(y[0]))

y_true,y_pred,y_scores = [],[],[]
for i in range(len(y_test)):
    y_true.append(y_test[i][1])
for i in range(len(y)):
    pred= y[i]
    y_scores.append(pred)
    if  pred>=0.2:
        y_pred.append(1)
    else:
        y_pred.append(0)

p, r, thresh = metrics.precision_recall_curve(y_true,y_scores)
pr_auc = metrics.auc(r, p)
auroc = metrics.roc_auc_score(y_true, y_scores)
print('auprc:',pr_auc)
print('auroc:',metrics.roc_auc_score(y_true, y_scores))
TN,FP,FN,TP = metrics.confusion_matrix(y_true, y_pred).ravel()
print('TN,FP,FN,TP:',TN,FP,FN,TP)
Accuracy = float(TP + TN) / float(TP + TN + FN + FP)
Specificity =float(TN) / float(TN + FP)
Sensitivity = float(TP) / float(TP + FN)
Precision = float(TP) / float(TP + FP)
Recall = float(TP) / float(TP + FN)
F1 = 2*float(TP)/float(2*TP+FP+FN)
MCC=float(TP*TN-FP*FN)/ math.sqrt(float(TP + FP) * float(TP + FN) * float(TN + FP) * float(TN + FN))
print('Acc:',Accuracy,' Sp:',Specificity,' Sn:',Sensitivity,'Pre',Precision,'Re',Recall,'F1',F1,' MCC:',MCC, 'auprc', pr_auc,'auroc:', auroc )
