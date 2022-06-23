# -*- coding: utf-8 -*-
from __future__ import division
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


# GBDT
def gbdt_grid_search(X_train, y_train, cv):
    parameters = {
        #'max_depth': [4, 6, 8],
        #'learning_rate': [0.5, 0.1, 0.01],
        #'n_estimators': [800, 1100, 1400]
        'max_depth': list(range(8, 12, 1)),
        'learning_rate': [0.01, 0.05, 0.001],
        'n_estimators': list(range(900, 1400, 100))
    }
    scorings = {'AUPRC': 'average_precision', 'f1': 'f1', 'ACC': 'accuracy', 'prec': 'precision', 'recall': 'recall',
                'AUROC': 'roc_auc'}
    clf = GradientBoostingClassifier(random_state=10)
    model = GridSearchCV(clf, param_grid=parameters, scoring=scorings, refit='AUROC', cv=cv, verbose=1, n_jobs=-1,
                         return_train_score=True)
    model.fit(X_train, y_train)
    return model


# LR
def lr_grid_search(X_train, y_train, cv):
    scorings = {'AUPRC': 'average_precision', 'f1': 'f1', 'ACC': 'accuracy', 'prec': 'precision', 'recall': 'recall',
                'AUROC': 'roc_auc'}
    parameters = {
        'C': [15, 10, 5, 1, 0.1, 0.01, 0.001, 0.0001],
        'penalty': ['l2', 'l1'],
    }
    clf = LogisticRegression(solver='liblinear')
    model = GridSearchCV(clf, scoring=scorings, param_grid=parameters, refit='AUROC', cv=cv, verbose=1, n_jobs=-1,
                         return_train_score=True)
    model.fit(X_train, y_train)
    return model


# Adaboost
def ab_grid_search(X_train, y_train, cv):
    parameters = {
        'learning_rate': [0.001, 0.01, 0.1, 0.5],
        #'n_estimators': [200, 300, 400, 500, 600, 700, 800]
        'n_estimators': list(range(1000, 2200, 100))
    }
    scorings = {'AUPRC': 'average_precision', 'f1': 'f1', 'ACC': 'accuracy', 'prec': 'precision', 'recall': 'recall',
                'AUROC': 'roc_auc'}
    clf = AdaBoostClassifier(n_estimators=100)
    model = GridSearchCV(clf, param_grid=parameters, scoring=scorings, refit='AUROC', cv=cv, verbose=1, n_jobs=-1,
                         return_train_score=True)
    model.fit(X_train, y_train)
    return model


# XGBoost
def xgb_grid_search(X_train, y_train, cv):
    parameters = {
        #'max_depth': [23, 24, 26],
        'learning_rate': [0.01, 0.05, 0.1],
        #'n_estimators': [3500, 3600, 3800]
        'max_depth': list(range(18, 22, 1)),
        'n_estimators': list(range(3300, 3800, 100))
    }
    scorings = {'AUPRC': 'average_precision', 'f1': 'f1', 'ACC': 'accuracy', 'prec': 'precision', 'recall': 'recall',
                'AUROC': 'roc_auc'}
    xlf = xgb.XGBClassifier(max_depth=10,
                            learning_rate=0.01,
                            n_estimators=2000)
    model = GridSearchCV(xlf, param_grid=parameters, scoring=scorings, refit='AUROC', cv=cv, verbose=1, n_jobs=-1,
                         return_train_score=True)
    model.fit(X_train, y_train)
    return model


# svm
def svm_grid_search(X_train, y_train, cv):
    param_grid = [{
        "gamma": [0.0001, 0.001],
        "C": [1000]}]
        #'C': [1000, 500, 250, 100, 50, 25, 1],
        #'gamma': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001]}]
    scorings = {'AUPRC': 'average_precision', 'f1': 'f1', 'ACC': 'accuracy', 'prec': 'precision', 'recall': 'recall',
                'AUROC': 'roc_auc'}
    svc = svm.SVC(kernel='rbf', probability=True)
    clf = GridSearchCV(svc, param_grid=param_grid, scoring=scorings, refit='AUROC', cv=cv, return_train_score=True,
                       n_jobs=-1)
    clf.fit(X_train, y_train)

    return clf


# RF
def rf_grid_search(X_train, y_train, cv):
    param_grid = {#'max_depth': [19, 20, 21, 22, 23],
                  #'n_estimators': [1100, 1200, 1300, 1400]}
                  'max_depth': list(range(20, 31, 1)),
                  'n_estimators': list(range(2000, 3100, 100))}
    scorings = {'AUPRC': 'average_precision', 'f1': 'f1', 'ACC': 'accuracy', 'prec': 'precision', 'recall': 'recall',
                'AUROC': 'roc_auc'}
    rfc = RandomForestClassifier(random_state=2019)
    model = GridSearchCV(rfc, param_grid=param_grid, scoring=scorings, refit='AUROC', cv=cv, return_train_score=True,
                       n_jobs=-1)
    model.fit(X_train, y_train)
    return model


# ERT
def ert_grid_search(X_train, y_train, cv):
    scorings = {'AUPRC': 'average_precision', 'f1': 'f1', 'ACC': 'accuracy', 'prec': 'precision', 'recall': 'recall',
                'AUROC': 'roc_auc'}
    parameters = {
        #'max_depth': [28, 29, 30, 31],
        #'n_estimators': [1400]
        'max_features': list(range(20, 35, 1)),
        'n_estimators': list(range(2000, 3500, 100))
    }
    xlf = ExtraTreesClassifier(max_depth=10, max_features=10, n_estimators=2000)
    model = GridSearchCV(xlf, param_grid=parameters, scoring=scorings, refit='AUROC', cv=cv, verbose=1, n_jobs=-1,
                                 return_train_score=True)
    model.fit(X_train, y_train)
    return model


# KNN
def knn_grid_search(X_train, y_train, cv):
    scorings = {'AUPRC': 'average_precision', 'f1': 'f1', 'ACC': 'accuracy', 'prec': 'precision', 'recall': 'recall',
                'AUROC': 'roc_auc'}
    parameters = {
        'n_neighbors': list(range(5, 15, 1)),
        #'p': [1]
    }
    knn = KNeighborsClassifier(n_neighbors=3)
    optimized_GBM = GridSearchCV(knn, param_grid=parameters, scoring=scorings, refit='AUROC', cv=cv, verbose=1, n_jobs=-1,
                                 return_train_score=True)
    optimized_GBM.fit(X_train, y_train)
    return optimized_GBM


# ANN
def ann_grid_search(X_train, y_train, cv):
    scorings = {'AUPRC': 'average_precision', 'f1': 'f1', 'ACC': 'accuracy', 'prec': 'precision', 'recall': 'recall',
                'AUROC': 'roc_auc'}
    parameters = {
        #'alpha': [4,5, 6] 
        'alpha': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    }
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
    model = GridSearchCV(clf, param_grid=parameters, scoring=scorings, refit='AUROC', cv=cv, verbose=1, n_jobs=-1,
                                 return_train_score=True)
    model.fit(X_train, y_train)
    return model
