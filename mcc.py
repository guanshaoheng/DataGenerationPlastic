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
from MCCUtil import ModifiedCamClay


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
[l, k, M, poisn, N] = [0.077, 0.01, 1.2, 0.3, 1.788]

# initial state
pc = 400.  # consolidation pressure 前期固结压力
p0 = 200.  # confining pressure 围压
# e0 = 0.68
# N = e0+l*np.log(pc_origin)+1.0
drainedFlag = True
driver = ModifiedCamClay(N=N, lambda_e=l, pc=pc,
         kappa_e=k, p0=p0, poisn=poisn, drainedFlag=drainedFlag, M=M)
driver.forward()

