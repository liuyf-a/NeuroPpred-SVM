import sys
import numpy as np
import warnings
import os
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
from sklearn import svm, datasets, metrics
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_curve, precision_recall_curve, precision_recall_fscore_support
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD as svd
import pickle, gzip
import math
from feature import *
from itertools import *


def readpeptides(posfile, negfile):  # return the peptides from input peptide list file
    posdata = open(posfile, 'r')
    pos = []
    for l in posdata.readlines():
        if l[0] == '>':
            continue
        else:
            pos.append(l.strip('\t0\n'))
    posdata.close()
    negdata = open(negfile, 'r')
    neg = []
    for l in negdata.readlines():
        if l[0] == '>':
            continue
        else:
            neg.append(l.strip('\t0\n'))
    negdata.close()
    return pos, neg


def combinefeature(pep, featurelist, dataset):
    a = np.empty([len(pep), 1])
    fname = []
    scaling = StandardScaler()
    # pca = svd(n_components=300)
    pca = PCA(0.99)
    vocab_name = []
    # print(a)
    if 'aap' in featurelist:
        aapdic = readAAP("./aap/aap_minmaxscaler_general.txt")
        f_aap = np.array([aap(pep, aapdic, 1)]).T
        a = np.column_stack((a, f_aap))
        # a = scaling.fit_transform(a)
        fname.append('AAP')
        # print(f_aap)
    if 'aat' in featurelist:
        aatdic = readAAT("./aat/aat_minmaxscaler_general.txt")
        f_aat = np.array([aat(pep, aatdic, 1)]).T
        a = np.column_stack((a, f_aat))
        # a = scaling.fit_transform(a)
        fname.append('AAT')
        # print(f_aat)
    if 'dpc' in featurelist:
        f_dpc, name = DPC(pep)
        # f_dpc = np.average(f_dpc, axis =1)
        a = np.column_stack((a, np.array(f_dpc)))
        fname = fname + name
    if 'aac' in featurelist:
        f_aac, name = AAC(pep)
        a = np.column_stack((a, np.array(f_aac)))
        fname = fname + name

    if 'paac' in featurelist:
        f_paac, name = PAAC(pep)
        # f_paac = pca.fit_transform(f_paac)
        a = np.column_stack((a, np.array(f_paac)))
        fname = fname + name

    if 'kmer' in featurelist:
        kmers = kmer(pep, 1)
        # f_kmer = np.array(kmers.X.toarray())
        f_kmer = np.array(kmers.X.toarray())
        vocab_name = kmers.vocab

        a = np.column_stack((a, f_kmer))
        fname = fname + ['kmer'] * len(f_kmer)

    if 'qso' in featurelist:
        f_qso, name = QSO(pep)
        # f_pa = pca.fit_transform(f_paac)
        a = np.column_stack((a, np.array(f_qso)))
        fname = fname + name

    if 'ctd' in featurelist:
        f_ctd, name = CTD(pep)
        a = np.column_stack((a, np.array(f_ctd)))
        fname = fname + name

    if 'bertfea' in featurelist:
        f_bertfea = np.array(bertfea('./bertfea/test/test_avg.txt'))
        a = np.column_stack((a, f_bertfea))
        fname = fname + ['bertfea'] * len(f_bertfea)

    if 'GGAP' in featurelist:
        f_ggap = np.array(GGAP(pep))
        print(f_ggap.shape)
        a = np.column_stack((a, f_ggap))
        fname = fname + ['GGAP'] * len(f_ggap)

    if 'ASDC' in featurelist:
        f_asdc = np.array(ASDC(pep))
        print(f_asdc.shape)
        a = np.column_stack((a, f_asdc))
        fname = fname + ['ASDC'] * len(f_asdc)

    if 'PSAAC' in featurelist:
        f_psaac = np.array(PSAAC(pep))
        print(f_psaac.shape)
        a = np.column_stack((a, f_psaac))
        fname = fname + ['PSAAC'] * len(f_psaac)

    return a[:, 1:], fname, vocab_name


def run_training(pos, neg, dataset):
    pep_combined = pos + neg
    #aap aat dpc aac kmer protvec paac qso ctd
    featurelist = ['PSAAC', 'ctd', 'aat', 'bertfea']
    print(featurelist)
    features, fname, vocab = combinefeature(pep_combined, featurelist, dataset) # 'aap', 'aat', 'aac'
    print(len(features[0]))
    '''for i in range(len(features)):
    	print(features[i])'''
    target = [1] * len(pos) + [0] * len(neg)
    train(features, target)

def train(features, target):
    with open('./model/svm-neuropeptides-avg-ctd-PSAAC-aat.pickle', 'rb') as fin:
        alldata = pickle.load(fin)
    print(alldata.keys())
    model1 = alldata['model']
    f_scaling = alldata['scaling']
    #f_scaling.fit(features)
    x_test = f_scaling.transform(features)
    y = np.array(target)
    y_pred = model1.predict(x_test)
    #y_scores = model1.decision_function(x_test)
    #print(y_scores)
    y_scores = model1.predict_proba(x_test)[:, 1]


    TN, FP, FN, TP = confusion_matrix(y, y_pred).ravel()
    print('TN, FP, FN, TP:', TN, FP, FN, TP)

    Specificity = TN / (TN + FP)
    ACC = float(TP + TN) / float(TP + FP + FN + TN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1Score = 2 * TP / (2 * TP + FP + FN)
    MCC = float(TP * TN - FP * FN) / math.sqrt(float(TP + FP) * float(TP + FN) * float(TN + FP) * float(TN + FN))

    p, r, thresh = metrics.precision_recall_curve(y, y_scores)
    pr_auc = metrics.auc(r, p)

    ro_auc = metrics.roc_auc_score(y, y_scores)

    print('Specificity:', Specificity, 'ACC:', ACC, 'Precision:', Precision, 'Recall:', Recall,
          'F1Score:', F1Score, 'MCC:', MCC, 'auprc:', pr_auc, 'auroc:', ro_auc)


if __name__ == "__main__":
    dataset = sys.argv[1]
    pos, neg = readpeptides("./datasets/"+dataset+"/test_pos.txt",
                            "./datasets/"+dataset+"/test_neg.txt")
    #print(pos, neg)
    run_training(pos, neg, dataset)
