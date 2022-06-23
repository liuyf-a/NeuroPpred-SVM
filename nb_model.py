import sys
import numpy as np
import warnings
import os

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold, LeaveOneOut, train_test_split, cross_val_score
from sklearn.decomposition import PCA
import pickle
import math
from feature import *


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
        f_bertfea = np.array(bertfea('./bertfea/train/train_avg.txt'))
        a = np.column_stack((a, f_bertfea))
        fname = fname + ['bertfea'] * len(f_bertfea)

    if 'fvs' in featurelist:
        f_bertfea = np.array(fvs('./FVs/neuropeptides/train_AAPIV.txt'))
        a = np.column_stack((a, f_bertfea))
        fname = fname + ['FVs'] * len(f_bertfea)

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

    if 'combin_fea' in featurelist:
        # f_combin_fea = np.array(combin_fea('./bertfea/protein/neuropeptides/train/train_avg.txt'))
        f_combin_fea = np.array(combin_fea('./combin_fea/neuropeptides/tr-PSAAC-ctd-aac-aap-aat-avg.txt'))
        a = np.column_stack((a, f_combin_fea))
        fname = fname + ['combin_fea'] * len(f_combin_fea)

    return a[:, 1:], fname, vocab_name


def nb_model(X_train, y_train):
    y_pred = []
    y_score = []
    y_true = []
    kf = StratifiedKFold(n_splits=5)
    for train_index, test_index in kf.split(X_train, y_train):
        x_train = X_train[train_index]
        tr_label = y_train[train_index]
        x_test = X_train[test_index]
        te_label = y_train[test_index]

        model = GaussianNB()
        model.fit(x_train, tr_label)
        pred = model.predict(x_test)
        score = model.predict_proba(x_test)[:, 1]
        y_pred.append(pred)
        y_score.append(score)
        y_true.append(te_label)
    return y_pred, y_score, y_true


def run_training(pos, neg, dataset):
    pep_combined = pos + neg
    pickle_info={}
    #print(pep_combined)
    # aap aat dpc aac kmer protvec paac qso ctd
    featurelist = ['PSAAC', 'ctd', 'aap', 'aat', 'bertfea']
    #featurelist = ['aap']
    print(featurelist)
    pickle_info['featurelist'] = featurelist
    features, fname, vocab = combinefeature(pep_combined, featurelist, dataset) # 'aap', 'aat', 'aac'
    print(len(features[0]))
    '''for i in range(len(features)):
    	print(features[i])'''
    pickle_info['feat_name'] = fname
    pickle_info['vocab'] = vocab
    #pickle.dump(features, open("features_latest.pickle", "wb"))
    #print(features)
    target = [1] * len(pos) + [0] * len(neg)
    #print(pep_combined)
    train(pep_combined, features, target, pickle_info, dataset)


def train(peptides, features, target, pickle_info, dataset):
    scaling = StandardScaler()
    scaling.fit(features)
    print(max(features[:,0]))
    x = scaling.transform(features)
    #print(max(x[:,1.txt]))
    y = np.array(target)

    pred, score, true = nb_model(x, y)

    preforence = []
    for i in range(0,5):
        y_true = np.array(true[i])
        y_pred = np.array(pred[i])
        y_score = np.array(score[i])
        TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
        print('TN, FP, FN, TP:', TN, FP, FN, TP)

        Specificity = TN / (TN + FP)
        ACC = float(TP + TN) / float(TP + FP + FN + TN)
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        F1Score = 2 * TP / (2 * TP + FP + FN)
        MCC = float(TP * TN - FP * FN) / math.sqrt(float(TP + FP) * float(TP + FN) * float(TN + FP) * float(TN + FN))

        p, r, thresh = metrics.precision_recall_curve(y_true, y_score)
        pr_auc = metrics.auc(r, p)

        ro_auc = metrics.roc_auc_score(y_true, y_score)
        print('第{}折评估指标：'.format(i))
        print('Specificity:', Specificity, 'ACC:', ACC, 'Precision:', Precision, 'Recall:', Recall,
              'F1Score:', F1Score, 'MCC:', MCC, 'auprc:', pr_auc, 'auroc:', ro_auc)
        
        all_scores = [Specificity, ACC, Precision, Recall, F1Score, MCC, pr_auc, ro_auc]
        preforence.append(all_scores)
    preforence = np.array(preforence).mean(axis=0)
    print("五折交叉验证平均")
    print(preforence)


if __name__ == "__main__":
    dataset = sys.argv[1]
    pos, neg = readpeptides("./datasets/"+dataset+"/train_pos.txt",
                            "./datasets/"+dataset+"/train_neg.txt")
    #print(pos, neg)
    run_training(pos, neg, dataset)
