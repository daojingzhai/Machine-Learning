#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 11:08:13 2018

@author: daojing
"""
from collections import defaultdict
from itertools import combinations
import csv
import os
import psutil
import time

class cached_property(object):
    """A cached property only computed once
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, obj, cls):
        if obj is None: return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value

class Base(object):
    """A base workflow for Apriori algorithm
    """

    def _before_generate_frequent_itemset(self):
        """Invoked before generate_frequent_itemset()
        """
        pass

    def _after_generate_frequent_itemset(self):
        """Invoked before generate_frequent_itemset()
        """
        pass

    def generate_frequent_itemset(self):
        """Generate and return frequent itemset
        """
        raise NotImplementedError("generate_frequent_itemset(self) need to be implemented.")

    def _before_generate_rule(self):
        """Invoked before generate_frequent_itemset()
        """
        pass

    def _after_generate_rule(self):
        """Invoked before generate_frequent_itemset()
        """
        pass

    def generate_rule(self):
        """Generate and return rule
        """
        raise NotImplementedError("generate_rule(self) need to be implemented.")

    def run(self):
        """Run Apriori algorithm and return rules
        """
        # generate frequent itemset
        self._before_generate_frequent_itemset()
        self.generate_frequent_itemset()
        self._after_generate_frequent_itemset()
        # generate rule
        self._before_generate_rule()
        self.generate_rule()
        self._after_generate_rule()

class Apriori(Base):
    """A simple implementation of Apriori algorithm
        Example:
        dataset = [
            ['bread', 'milk'],
            ['bread', 'diaper', 'beer', 'egg'],
            ['milk', 'diaper', 'beer', 'cola'],
            ['bread', 'milk', 'diaper', 'beer'],
            ['bread', 'milk', 'diaper', 'cola'],
        ]
        minsup = minconf = 0.6
        apriori = Apriori(dataset, minsup, minconf)
        apriori.run()
        apriori.print_rule()
        Results:
            Rules
            milk --> bread (confidence = 0.75)
            bread --> milk (confidence = 0.75)
            diaper --> bread (confidence = 0.75)
            bread --> diaper (confidence = 0.75)
            beer --> diaper (confidence = 1.0)
            diaper --> beer (confidence = 0.75)
            diaper --> milk (confidence = 0.75)
            milk --> diaper (confidence = 0.75)
    """

    def __init__(self, transaction_list, minsup, minconf, selected_items=None):
        """Initialization
        :param transaction_list: a list cantains transaction
        :param minsup: minimum support
        :param minconf: minimum confidence
        :param selected_items: selected items in frequent itemset, default `None`
        """
        self.transaction_list = transaction_list
        self.transaction_list_full_length = len(transaction_list)
        self.minsup = minsup
        self.minconf = minconf
        if selected_items is not None and selected_items is not []:
            self.selected_items = frozenset(selected_items)
        else:
            self.selected_items = None

        self.frequent_itemset = dict()
        # support for every frequent itemset
        self.frequent_itemset_support = defaultdict(float)
        # convert transaction_list
        self.transaction_list = list([frozenset(transaction) \
                                      for transaction in transaction_list])

        self.rule = []

    def set_selected_items(self, selected_items):
        """Set selected items
        """
        self.selected_items = frozenset(selected_items)

    @cached_property
    def items(self):
        """Return all items in the self.transaction_list
        """
        items = set()
        for transaction in self.transaction_list:
            for item in transaction:
                items.add(item)
        return items

    def filter_with_minsup(self, itemsets):
        """Return subset of itemsets which satisfies minsup
        and record their support
        """
        local_counter = defaultdict(int)
        for itemset in itemsets:
            for transaction in self.transaction_list:
                if itemset.issubset(transaction):
                    local_counter[itemset] += 1
        # filter with counter
        result = set()
        for itemset, count in local_counter.items():
            support = float(count) / self.transaction_list_full_length
            if support >= self.minsup:
                result.add(itemset)
                self.frequent_itemset_support[itemset] = support
        return result

    def _after_generate_frequent_itemset(self):
        """Filter frequent itemset with selected items
        """
        if self.selected_items is None:
            return
        local_remove = []
        for key, val in self.frequent_itemset.items():
            for itemset in val:
                if not self.selected_items.issubset(itemset):
                    local_remove.append((key, itemset))
        for (key, itemset) in local_remove:
            self.frequent_itemset[key].remove(itemset)

    def generate_frequent_itemset(self):
        """Generate and return frequent itemset
        """

        def _apriori_gen(itemset, length):
            """Return candidate itemset with given itemset and length
            """
            # simply use F(k-1) x F(k-1) (itemset + itemset)
            return set([x.union(y) for x in itemset for y in itemset \
                        if len(x.union(y)) == length])

        k = 1
        current_itemset = set()
        """ generate 1-frequnt_itemset
        """
        for item in self.items: current_itemset.add(frozenset([item]))
        self.frequent_itemset[k] = self.filter_with_minsup(current_itemset)
        """generate k-frequent_itemset
        """
        while True:
            k += 1
            current_itemset = _apriori_gen(current_itemset, k)
            current_itemset = self.filter_with_minsup(current_itemset)
            if current_itemset != set([]):
                self.frequent_itemset[k] = current_itemset
            else:
                break
        return self.frequent_itemset

    def _generate_rule(self, itemset, frequent_itemset_k):
        """Generate rule with F(k) in DFS style
        """
        # make sure the itemset has at least two element to generate the rule
        if len(itemset) < 2:
            return
        for element in combinations(list(itemset), 1):
            rule_head = itemset - frozenset(element)
            confidence = self.frequent_itemset_support[frequent_itemset_k] / \
                         self.frequent_itemset_support[rule_head]
            if confidence >= self.minconf:
                rule = ((rule_head, itemset - rule_head), confidence)
                # if rule not in self.rule, add and recall _generate_rule() in DFS
                if rule not in self.rule:
                    self.rule.append(rule);
                    self._generate_rule(rule_head, frequent_itemset_k)

    def generate_rule(self):
        """Generate and return rule
        """
        # generate frequent itemset if not generated
        if len(self.frequent_itemset) == 0:
            self.generate_frequent_itemset()
        # generate in DFS style
        for key, val in self.frequent_itemset.items():
            if key == 1:
                continue
            for itemset in val:
                self._generate_rule(itemset, itemset)
        return self.rule

    def print_frequent_itemset(self):
        """Print out frequent itemset
        """
        sum = 0

        print('Patterns:======================================================\n')
        for key, val in self.frequent_itemset.items():
            # stdout.write('frequent itemset size of {0}:\n'.format(key))
            for itemset in val:
                print('(',', '.join(itemset),')',round(self.frequent_itemset_support[itemset], 3))
                sum = sum + 1
        print('======================================================\n')
        return sum


    def print_rule(self):
        """Print out rules
        """
        sum = 0
        print('Patterns:======================================================\n')
        for rule in self.rule:
            head = rule[0][0]
            tail = rule[0][1]
            confidence = rule[1]
            print('(',', '.join(head),')((',', '.join(tail),'),',round(confidence, 3),')')
            sum = sum + 1
        print('======================================================\n')
        return sum


def preprocess():
    with open('Groceries.csv', 'r', encoding='utf-8') as csv_file:
        dataset = []
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for line in csv_reader:
            dataset.append(line[1].strip('{').strip('}').split(','))
        return dataset
    
if __name__ == '__main__':
    ts = time.time()
    minsup = 0.05
    minconf = 0.05
    ap = Apriori(preprocess(), minsup, minconf)
    # run algorithm
    ap.run()
    # print out frequent itemsets and rules
    sum_itemset = ap.print_frequent_itemset()
    sum_rules = ap.print_rule()
    te = time.time()
    process = psutil.Process(os.getpid())
    print("Memory cost: ",process.memory_info().rss/1024/1024,"MB")
    print("Time cost: ",te-ts,"s")
    print("Apriori algorithm returned",sum_itemset,sum_rules)