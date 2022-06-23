import sys
import numpy as np
import warnings
import os
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold, LeaveOneOut, train_test_split
from sklearn.decomposition import PCA, TruncatedSVD as svd
import pickle
import math
from feature import *
from ML_grid_search_model import *


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
    pickle_info={}
    #print(pep_combined)
    # aap aat dpc aac kmer protvec paac qso ctd
    featurelist = ['ctd', 'aat', 'PSAAC', 'bertfea']
    print(featurelist)
    pickle_info['featurelist'] = featurelist
    features, fname, vocab = combinefeature(pep_combined, featurelist, dataset) # 'aap', 'aat', 'aac'
    print(len(features[0]))
    '''for i in range(len(features)):
    	print(features[i])'''
    pickle_info['feat_name'] = fname
    pickle_info['vocab'] = vocab
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
    cv = StratifiedKFold(n_splits=5)

    # 切换模型
    model = svm_grid_search(x, y, cv)
    aapdic = readAAP("./aap/aap_minmaxscaler_general.txt")
    aatdic = readAAT("./aat/aat_minmaxscaler_general.txt")
    pickle_info ['aap'] = aapdic
    pickle_info ['aat'] = aatdic
    pickle_info ['scaling'] = scaling
    pickle_info ['model'] = model
    pickle_info ['training_features'] = features
    pickle_info ['training_targets'] = y
    pickle.dump(pickle_info, open("./model/svm-"+dataset+"-avg-ctd-PSAAC-aat.pickle", "wb"))
    print("Best parameters: ", model.best_params_)
    print("Best accuracy: :", model.best_score_)

    cv_accracy = model.cv_results_['mean_test_ACC'][model.best_index_]
    cv_auprc = model.cv_results_['mean_test_AUPRC'][model.best_index_]
    cv_precision = model.cv_results_['mean_test_prec'][model.best_index_]
    cv_recall = model.cv_results_['mean_test_recall'][model.best_index_]
    cv_auroc = model.cv_results_['mean_test_AUROC'][model.best_index_]
    cv_f1 = model.cv_results_['mean_test_f1'][model.best_index_]

    y_train_t=y.tolist()
    y_train_t.count(1)
    y_train_t.count(0)
    TP1=y_train_t.count(1)*cv_recall
    FP1=(TP1/cv_precision)-TP1
    TN1=y_train_t.count(0)-FP1
    FN1=y_train_t.count(1)-TP1
    print('TP:',TP1,',TN:',TN1,',FP:',FP1,',FN:',FN1)
    cv_specificity = Specificity=TN1/(TN1+FP1)

    if ((float(TP1 + FP1) * float(TN1 + FN1)) != 0):
        cv_MCC = float(TP1*TN1-FP1*FN1)/ math.sqrt(float(TP1 + FP1) * float(TP1 + FN1) * float(TN1 + FP1) * float(TN1 + FN1))
        print('Specificity_train:',cv_specificity,',ACC_train:',cv_accracy,',Precision_train:',cv_precision,',Recall_train:',cv_recall,',F1Score_train:',cv_f1,',MCC_train:',cv_MCC,',auprc_train:',cv_auprc,',auroc_train:',cv_auroc)
    else:
        print('Specificity_train,ACC_train,Precision_train,Recall_train,F1Score_train,auprc_train,auroc_train:',
              cv_specificity,cv_accracy,cv_precision,cv_recall,cv_f1,cv_auprc,cv_auroc)



if __name__ == "__main__":
    dataset = sys.argv[1]
    pos, neg = readpeptides("./datasets/"+dataset+"/train_pos.txt",
                            "./datasets/"+dataset+"/train_neg.txt")
    #print(pos, neg)
    run_training(pos, neg, dataset)
