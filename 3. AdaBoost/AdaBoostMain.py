#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Created on Sat Jun 16 20:02:09 2018
    
    @author: daojing
    """

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score 



""" Get Error Rate """
def get_error_rate(pred, Y):
    return sum(pred != Y)/float(len(Y))

""" Print Error Rate """
def print_error_rate(err):
    print ('Error rate: Training: %.4f - Test: %.4f' % err)

""" Generic Classifier """
def generic_clf(Y_train, X_train, Y_test, X_test, clf):
    clf.fit(X_train,Y_train)
    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)
    return get_error_rate(pred_train, Y_train), get_error_rate(pred_test, Y_test)


""" Adamboost Algorithm """
def adaboost_clf(Y_train, X_train, Y_test, X_test, M, clf):
    n_train, n_test = len(X_train), len(X_test)
    
    # Initialize weights
    w = np.ones(n_train) / n_train
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]
    alpha = []
    CLF = []
    
    for i in range(M):
        # Fit a classifier with the specific weights
        clf.fit(X_train, Y_train, sample_weight = w)
        pred_train_i = clf.predict(X_train)
        pred_test_i = clf.predict(X_test)
        # Indicator function
        miss = [int(x) for x in (pred_train_i != Y_train)]
        # Equivalent with 1/-1 to update weights
        miss2 = [x if x==1 else -1 for x in miss]
        # Error
        err_m = np.dot(w,miss) / sum(w)
        # Alpha
        alpha_m = 0.5 * np.log( (1 - err_m) / float(err_m))
        # New weights
        w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))
        # Add to prediction
        pred_train = [sum(x) for x in zip(pred_train,
                                          [x * alpha_m for x in pred_train_i])]
            pred_test = [sum(x) for x in zip(pred_test,
                                             [x * alpha_m for x in pred_test_i])]
                
                alpha.append(alpha_m)
                CLF.append(clf)
    
    pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)
    train_AUC = roc_auc_score(Y_train,preåd_train)
    test_AUC = roc_auc_score(Y_test,pred_test)
    # Return error rate in train and test set
    return get_error_rate(pred_train, Y_train), get_error_rate(pred_test, Y_test),train_AUC,test_AUC,alpha,CLF


""" Plot Function """

def plot_error_rate(er_train, er_test, x_range):
    plt.title('Error Analysis')
    plt.plot(x_range, er_test, color='darkblue', label='Test', linewidth=2)
    plt.plot(x_range, er_train, color='skyblue', label='Train', linewidth=2)
    plt.legend()
    plt.axhline(y=er_test[0], linewidth=1, color = 'red', ls = 'dashed')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Error')
    plt.show()



def plot_auc_rate(auc_train, auc_test, x_range):
    plt.title('AUC Analysis')
    plt.plot(x_range, auc_test, color='darkblue', label='Test', linewidth=2)
    plt.plot(x_range, auc_train, color='skyblue', label='Train', linewidth=2)
    plt.legend()
    plt.axhline(y=auc_test[0], linewidth=1, color = 'red', ls = 'dashed')
    plt.xlabel('Number of Iterations')
    plt.ylabel('AUC')
    plt.show()

""" Modify Label Function """

def modified_label(label):
    if np.min(label) == 0:
        label[label==0] = -1
    return label



""" Main Script """
if __name__ == '__main__':
    
    # Read data
    Path = '/Users/daojing/Desktop/HW5/adult_dataset/'
    X_train = np.loadtxt(Path+'adult_train_feature.txt')
    Y_train = np.loadtxt(Path+'adult_train_label.txt')
    X_test = np.loadtxt(Path+'adult_test_feature.txt')
    Y_test = np.loadtxt(Path+'adult_test_label.txt')
    
    # transform the label into [-1, 1]
    
    Y_train = modified_label(Y_train)
    Y_test = modified_label(Y_test)
    
    # Fit a simple decision tree first
    clf_tree = DecisionTreeClassifier(max_depth = 1, random_state = 1)
    er_tree = generic_clf(Y_train, X_train, Y_test, X_test, clf_tree)
    
    # Fit Adaboost classifier using a decision tree as base estimator
    # Test with different number of iterations
    er_train, er_test = [], []
    auc_train, auc_test = [],[]
    x_range = range(1, 400, 3)
    
    for i in x_range:
        print("i = {}".format(i))
        er_i = adaboost_clf(Y_train, X_train, Y_test, X_test, i, clf_tree)
        er_train.append(er_i[0])
        er_test.append(er_i[1])
        auc_train.append(er_i[2])
        auc_test.append(er_i[3])
    
    
    # Plot error rate & AUC vs number of iterations
    plot_error_rate(er_train, er_test, x_range)
    plot_auc_rate(auc_train, auc_test, x_range)
