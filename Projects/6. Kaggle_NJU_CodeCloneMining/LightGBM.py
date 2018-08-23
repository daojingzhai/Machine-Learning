#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 22:02:20 2018

@author: daojing
"""

# coding: utf-8
# pylint: disable = invalid-name, C0111


# load or create your dataset
print('Load data...')
df_train = pd.read_csv('/Users/daojing/Downloads/datatrain', sep='\t').values
df_test = pd.read_csv('/Users/daojing/Downloads/datatest', sep='\t').values
y_train = pd.read_csv('/Users/daojing/Downloads/labeltrain', sep='\t').values
y_test = pd.read_csv('/Users/daojing/Downloads/labeltest', sep='\t').values


# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)


# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'auc'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)

print('Save model...')
# save model to file
gbm.save_model('model.txt')

print('Start predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# eval
print('The rmse of prediction is:', f1_score(y_test, y_pred) ** 0.5)