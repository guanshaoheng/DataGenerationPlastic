# -*- coding: utf-8 -*-
"""
Created on Wed May 12 13:37:09 2021
Triaxial test cases [deviatoric hardening(DH) model]
Generating stress-strain sequence via DH model

@author: Qu Tongming

Note: Tensile normal stress is positive
"""

import numpy as np  # import module
import pandas as pd
import glob, os
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'
# ------------------------------------------------

# material data
RF = 1.331  # failure slope; slope of the failure envelope
RC = 1.215  # critical slope; the slope of zero dilatancy line
A = 0.011  # hardening parameters; the constant A which appears in the hardening function

DP = 0.3  # Increments of mean effective stress(stress-dominated loading required)
DQ = 0.9  # Increments of deviatoric stress increment(stress-dominated loading required)
# ------------------------------------------------
# Loop and integral
Number_iteration = 400

data1 = pd.DataFrame({'p': [1],
                      'deviatoric_stress': [2],
                      "strain": [3],
                      "volume": [4],
                      "item1": [5]})

df = []
# 假设最大轴向应变为 0.1,常规三轴压缩 CTC
jj = 0
x_strain = []
z_strain = []
P11 = []
Volume = []
data1 = []
DEV = []
DEQ = []
P1 = 50


def numerical_integration(DP, DQ, Number_iteration=2500):  # the use of keyword argument
    EV = 0
    EQ = 0
    Mean_P = []
    Devi_q = []
    Devi_strain = []
    Volume = []
    global P
    Q = 0
    global BM, RF, RC, G, A, HP, HE
    global R0
    for i in range(Number_iteration):
        G = 3719.81018 * P ** 0.18  # shear modulus
        BM = 6720.74398 * P  # bulk modulus
        R0 = Q / P  # 实时关系 用来判断什么时候停
        FP = -R0  # f是屈服函数  df/dp= 3.61
        FQ = 1.  # 3.61中对q分别求导
        QP = RC - R0  # 公式 3.65
        QQ = 1.  # 第一个q是3.64对q求导
        if R0 > RF:
            break
        if EQ > 0.25:
            break
        # Stan's hardening      
        HP = P * (RF - R0) ** 2 / (A * RF)  # 3.66 --可从3.57第二项里面推导；plastic hardening modulus hp
        HE = BM * FP * QP + 3 * G * FQ * QQ  # 3.57的第一项
        # strain dominated loading
        D = np.array([[BM - (BM * BM * FP * QP) / (HE + HP), -(3 * G * BM * QP * FQ) / (HE + HP)],
                      [-(3 * G * BM * QQ * FP) / (HE + HP), 3 * G - 9 * G * G * QQ * FQ / (HE + HP)]])
        # stress dominated loading
        DET = D[0][0] * D[1][1] - D[0][1] * D[1][0]  # 行列式
        C = [[1, 1], [1, 1]]
        C[0][0] = D[1][1] / DET  # 应力加载
        C[0][1] = -D[0][1] / DET
        C[1][0] = -D[1][0] / DET
        C[1][1] = D[0][0] / DET
        # 累积
        dEV = C[0][0] * DP + C[0][1] * DQ  # Increments of volumetrical strain
        dEQ = C[1][0] * DP + C[1][1] * DQ  # Increments of deviatoric plastic strain
        EV = EV + dEV  # 累积循环
        EQ = EQ + dEQ
        P = P + DP
        Q = Q + DQ
        # Store data
        Mean_P.append(P)
        Devi_q.append(Q)
        Devi_strain.append(EQ)
        Volume.append(EV)
        DEV.append(dEV)
        DEQ.append(dEQ)
        R0 = Q / P  # 更新   判断
    # converting list to dataframe and then concatinate dataframe
    Mean_P1 = pd.DataFrame(Mean_P)
    Devi_q1 = pd.DataFrame(Devi_q)
    Devi_strain1 = pd.DataFrame(Devi_strain)
    Volume1 = pd.DataFrame(Volume)
    data = pd.concat([Mean_P1, Devi_q1, Devi_strain1, Volume1], axis=1)  # The axis（0 or 1）to concatenate along.
    names = ['p', 'deviatoric_stress', 'deviatoric_strain', 'volume']
    data.columns = names

    return data


def Generate_stress_strain_pairs():
    # 创建一个空的 DataFrame
    data2 = pd.DataFrame(columns=['p', 'deviatoric_stress', "deviatoric_strain", "volume", 'case'])
    global P
    for ii in range(475):
        P1 = 50  # initial mean effective stress
        P = P1 + ii * 2
        P0 = P1 + ii * 2
        data1 = numerical_integration(DP, DQ, Number_iteration=2500)
        data1['confining_stress'] = P0
        global jj
        jj = jj + 1
        data1['case'] = jj
        data2 = pd.concat([data2, data1], axis=0)  # （ axis = 0，列对齐（增加行））
    return data2


new_data = Generate_stress_strain_pairs()  # Call function

new_data['Deviat_plastic_strain'] = new_data.apply(
    lambda x: (x['deviatoric_strain'] - x['deviatoric_stress'] / (3 * G)), axis=1)

new_data = new_data.drop(new_data.index[0], axis=0)
order = ['case', 'Deviat_plastic_strain', "deviatoric_strain", 'deviatoric_stress', "volume", 'p', 'confining_stress']
new_data = new_data[order]

##  write csv file
new_data.to_csv('synthesis_data.csv', sep=',', header=True, index=True)  # , names=["","q","strain","volume"]

fig, ax = plt.subplots()
ax_plot = plt.scatter(new_data['deviatoric_strain'], new_data['deviatoric_stress'], c=new_data['confining_stress'],
                      cmap=plt.get_cmap('coolwarm'), s=5, alpha=0.5, linewidth=0, label='q')  # coolwarm
plt.xlabel('Deviatoric strain', fontsize=14)
plt.ylabel('Deviatoric stress (kPa)', fontsize=14)
plt.xlim(0, 0.3)
plt.ylim(0, 2250)
ax.set_xticks([0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
# 加上颜色棒
fig.colorbar(ax_plot)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig("DH_stress_strain_pairs.png", dpi=600, bbox_inches="tight")
plt.show()

fig2, ax2 = plt.subplots()
ax_plot = plt.scatter(new_data['deviatoric_strain'], new_data['volume'], c=new_data['confining_stress'],
                      cmap=plt.get_cmap('jet'), s=5, alpha=0.5, linewidth=0, label='q')  # coolwarm
plt.xlabel('Deviatoric strain', fontsize=14)
plt.ylabel('Volumetric strain', fontsize=14)
plt.xlim(0, 0.3)
plt.ylim(0, 0.025)
# ax.set_xticks([0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
# ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
# 加上颜色棒
fig.colorbar(ax_plot)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig("DH_volumetric_p.png", dpi=600, bbox_inches="tight")
plt.tight_layout()
plt.show()
