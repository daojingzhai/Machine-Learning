#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 02:56:48 2018

@author: daojing
"""

import os
from pycparser import parse_file


path = "/Users/daojing/Desktop/DataMiningLAB/train"


DirList = os.listdir(path)
AST = {}

for d in DirList:
    DirPath = os.path.join(path, d)
    ast = []
    for f in os.listdir(DirPath):
        filePath = os. path.join(DirPath, f)
        if os.path.isfile(filePath):
            file = open(filePath)
            tmp_ast = parse_file(filePath, use_cpp=True)
            ast.append(tmp_ast)

            file.close()
    print("dir: " + d + " is done...")
    AST[d] = ast

