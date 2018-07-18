#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Created on Mon Jun 18 21:15:13 2018
    
    @author: daojing
    """


import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt



class RandomForest:
    
    """ Initial """
    def __init__(self, Base_Model, k, T=100):
        self.Model = Base_Model  # Base_Model is a class not an object
        self.T = T
        self.k = k
        self.params = {
            'h': [],
            'sample_index': [],
            'feature_index': []
        }
        self.__score__ = 0.0
    
    
    
    def model_training(self, Data, label):
        model = self.Model().fit(Data, label)
        return model
    
    
    
    def generate_index(self, n, k, bootstrap=False):
        index = np.random.choice(n, k, replace=bootstrap)
        return index
    
    
    
    def predict(self, feature):
        pre = np.zeros(feature.shape[0])
        length = len(self.params['h'])
        for i in range(length):
            tmp_fea = feature[:, self.params['feature_index'][i]]
            pre += self.params['h'][i].predict(tmp_fea)
        return np.sign(pre)
    
    
    
    def fit(self, Data, label, sample_weight=None):
        h = []
        s_index = []
        f_index = []
        
        for t in range(self.T):
            # bootstrap step
            sample_index = self.generate_index(Data.shape[0], label.shape[0], True)
            # feature selection step
            feature_index = self.generate_index(Data.shape[1], self.k, False)
            modified_data = Data[sample_index, :][:, feature_index]
            modified_label = label[sample_index]
            
            h.append(self.model_training(modified_data, modified_label))
            s_index.append(sample_index)
            f_index.append(feature_index)
        
        self.params = {
            'h': h,
            'sample_index': s_index,
            'feature_index': f_index
        }
        # Finished building the model...
        
        # predict out of bag
        removal_data = np.delete(Data, sample_index, axis=0)
        removal_label = np.delete(label, sample_index, axis=0)
        self.__score__ = self.score(removal_data, removal_label, sample_weight)
    
    def score(self, X, y, sample_weight=None):
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y, self.predict(X), sample_weight=sample_weight)



def cross_validation(model, feature, label):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5)  
    i = 0
    final_score = 0
    for train_index, test_index in kf.split(feature):
        X_train, X_test = feature[train_index], feature[test_index]
        y_train, y_test = label[train_index], label[test_index]
        
        model.fit(X_train, y_train)
        if not isinstance(model, RandomForest):
            from sklearn.metrics import roc_auc_score
            score = roc_auc_score(y_test, model.predict(X_test))
        else:
            score = model.score(X_test, y_test)
        
        i += 1
        final_score += score
    
    return final_score / kf.get_n_splits()



""" Modify Label Function """

def modify_label(label):
    if np.min(label) == 0:
        label[label == 0] = -1
    return label    


""" Plot Function """

def plot_auc_rate(auc_train, auc_test, x_range):
    plt.title('AUC Analysis')
    plt.plot(x_range, auc_test, color='darkblue', label='Test', linewidth=2)
    plt.plot(x_range, auc_train, color='skyblue', label='Train', linewidth=2)
    plt.legend() 
    plt.axhline(y=auc_test[0], linewidth=1, color = 'red', ls = 'dashed')
    plt.xlabel('Number of Iterations')
    plt.ylabel('AUC')
    plt.show()


""" Main Script """
if __name__ == "__main__":
    
    
    # Read data
    Path = '/Users/daojing/Desktop/HW5/adult_dataset/'
    train_feature = np.loadtxt(Path+'adult_train_feature.txt')
    train_label = np.loadtxt(Path+'adult_train_label.txt')
    test_feature = np.loadtxt(Path+'adult_test_feature.txt')
    test_label = np.loadtxt(Path+'adult_test_label.txt')
    
    
    # transform the label into [-1, 1]
    train_label = modify_label(train_label)
    test_label = modify_label(test_label)
    
    
    # choose classification Class:
    CLF = DecisionTreeClassifier
    
    x_range = range(1,400,3)
    
    
    # Parameter: random choose k features
    k = 5
    best_T = 0
    best_cv = 0.0
    auc_train, auc_test = [],[]
    
    
    i = 0
    for T in x_range:
        print("T = {}".format(T))
        rf = RandomForest(CLF, k, T)
        auc_train.append(cross_validation(rf, train_feature, train_label))
        auc_train[i] = cross_validation(rf, train_feature, train_label)
        print("CV score: " + str(auc_train[i]))
        score = rf.score(test_feature, test_label)
        auc_test.append(score)
        if auc_train[i] > best_cv:
            best_cv = auc_train[i]
            best_T = T
        
        i = i+1
    
    
    # Test
    rf = RandomForest(CLF, k, best_T)
    rf.fit(train_feature, train_label)
    test_score = rf.score(test_feature, test_label)
    print("=====================================")
    print("AUC Score of Test Dataset: {}".format(test_score))
    plot_auc_rate(auc_train, auc_test, x_range)

