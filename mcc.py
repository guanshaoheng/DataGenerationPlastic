#!/usr/bin/env python
# encoding: utf-8
"""
@file: modified_cam_clay_model.py
@time: 2019/11/9 21:36
@email: guanshaoheng@qq.com
@application：
                 1.根据修正剑桥模型计算应力应变
                 2.常规三轴(drained & undrained)
"""

import numpy as np
from matplotlib import pyplot as plt
import os
import math
from MCCUtil import ModifiedCamClay, loadingPathReader


# initiation
# material properties
'''
l: the slope of the e-lnp line
k: the slope of the unload e-lnp line
N: the y coordinate of the natural consolidation line (e-lnp)
M: the q/p ratio at the critical state
poisn: the poisn ratio
'''

# [l, k, M, poisn, N] = [0.2, 0.04, 0.95, 0.15, 2.5]
[l, k, M, poisn, N] = [0.077, 0.04, 1.2, 0.3, 1.788]

# initial state
pc = 200  # consolidation pressure 前期固结压力
p0 = 201.  # confining pressure 围压

# --------------------------------------------------
# Debug && baseline
# loadMode = 'drained'  # 'drained' 'undrained'
# driver = ModifiedCamClay(N=N, lambda_e=l, pc=pc,
#          kappa_e=k, p0=p0, poisn=poisn, loadMode=loadMode, M=M, dimensionNum=2)
# driver.forward()

# --------------------------------------------------
# Application
loadPathList = loadingPathReader()
loadMode = 'random'
for i, path in enumerate(loadPathList):
    driver = ModifiedCamClay(N=N, lambda_e=l, pc=pc,
                             kappa_e=k, p0=p0, poisn=poisn, loadMode=loadMode, M=M, dimensionNum=2)
    driver.forward(numIndex=i, path=path)