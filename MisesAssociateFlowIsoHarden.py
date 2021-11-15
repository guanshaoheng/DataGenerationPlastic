import os
import numpy as np
import matplotlib.pyplot as plt
from MCCUtil import plotSubFigures, loadingPathReader
from plotConfiguration2D import plotConfiguration2D

"""
        The constitutive model is under the elastoplastic 
        framework of Mises yield function, associated-flow 
        rule, and iso-hardening.
        
        Gaussian process engaged for random loading path 
        generation.
        
        This script is used to generate datasets for phys-
        ics-constrained constitutive network training.
         
        Author: Shaoheng Guan
        Email:  shaohengguan@gmail.com

        Reference:
            [1] 
"""


class MisesAssociateFlowIsoHarden:
    def __init__(self, loadMode='axial'):
        # ---------------------------------------------------
        # material parameters
        self.youngsModulus = 200e9
        self.poisson = 0.3
        self.A = 500e6
        self.n = 0.2
        self.epsilon0 = 0.05
        self.yieldStress = 200e6
        self.hardening = self.getHardening(epsPlastic=0.)
        self.D = self.tangentAssemble(
            lam=self.youngsModulus * self.poisson / (1 + self.poisson) / (1 - 2 * self.poisson),
            G=self.youngsModulus / 2 / (1 + self.poisson))

        # ---------------------------------------------------
        # state variables [stress and strain vector in voigt notion]
        self.sig = np.zeros(3)
        self.vonMises = self.getVonMises(self.sig)
        self.eps = np.zeros(3)
        self.epsPlastic = 0.
        self.epsPlasticVector = np.zeros(3)
        self.sigTrial = np.zeros(3)
        self.lastYield = -1
        self.yieldValue = self.yieldFunction(self.sig)
        self.loadHistoryList = [np.array(list(self.sig) + list(self.eps) +
                                         [self.vonMises, self.epsPlastic, self.hardening] +
                                         list(self.epsPlasticVector) + [self.yieldValue, 0])]

        # ---------------------------------------------------
        # load configuration
        self.loadMode = loadMode  # 'axial' or 'random'
        if self.loadMode == 'random':
            self.epsAxialObject = 0.004  # random loading
        else:
            self.epsAxialObject = 0.01
        self.iterationNum = int(1e2)
        self.depsAxial = self.epsAxialObject / self.iterationNum

        # ---------------------------------------------------
        # Tolerance
        self.yieldTolerance = 1e1

    def forward(self, path=None, sampleIndex=None):
        if self.loadMode == 'random':
            self.iterationNum = len(path)
        for i in range(self.iterationNum):
            if self.loadMode == 'random':
                if i == 0:
                    deps = path[0]
                else:
                    deps = (path[i] - path[i - 1]) * self.epsAxialObject / np.max(np.abs(path))
            else:
                deps = self.getAxialDeps()
                if 0.3 * self.iterationNum < i < 0.65 * self.iterationNum:
                    deps = -deps
            self.sigTrial = self.sig + np.dot(self.D, deps)
            self.vonMises = self.getVonMises(self.sigTrial)
            self.hardening = self.getHardening(self.epsPlastic)
            yieldValue = self.yieldFunction(self.sigTrial)
            iteration = 0

            if yieldValue <= 0:  # elastic
                self.sig = self.sigTrial
                self.lastYield = yieldValue
                deps_plastic = np.zeros(3)
            elif self.lastYield < -self.yieldTolerance:  # plastic and last step is elastic
                r_mid, yield_mid = self.transiformationSplit(deps)
                self.sig = self.sig + np.dot(self.D, r_mid * deps)
                self.vonMises = self.getVonMises(self.sig)
                self.eps += r_mid * deps
                self.yieldValue = yield_mid
                self.loadHistoryList.append(np.array(list(self.sig) + list(self.eps) +
                                                     [self.vonMises, self.epsPlastic, self.hardening] +
                                                     list(self.epsPlasticVector) + [self.yieldValue, iteration]))
                deps = (1 - r_mid) * deps
                self.lastYield = yield_mid
                yieldValue = yield_mid
                # update the trial stress
                self.sigTrial = self.sig + np.dot(self.D, deps)
                self.vonMises = self.getVonMises(self.sigTrial)

            if yieldValue <= 0:
                pass
            else:  # last step is plastic
                # print('plastic')
                sigTrial = self.sigTrial
                mises = self.getVonMises(sigTrial)
                epsPlastic = self.epsPlastic
                while yieldValue > 0:
                    self.dFdS, self.dFdEps_p = self.getDiffVectorOfYieldFunction(sig=sigTrial, mises=mises, epsPlastic=epsPlastic)
                    dfds_mat = self.dFdS.reshape([3, 1])
                    h = -self.dFdEps_p * np.sqrt(2 / 3 * (dfds_mat.T @ dfds_mat))[0, 0]
                    H = (h + dfds_mat.T @ self.D @ dfds_mat)[0, 0]
                    dLambda = (dfds_mat.T @ self.D @ deps.reshape([-1, 1]) / H)[0, 0]
                    deps_plastic = dLambda * dfds_mat
                    epsPlastic = self.epsPlastic + np.sqrt(deps_plastic.T @ deps_plastic)[0, 0]
                    self.hardening = self.getHardening(epsPlastic)
                    sigTrial = self.sigTrial - np.dot(self.D, deps_plastic).reshape(-1)
                    mises = self.getVonMises(sigTrial)
                    yieldValue = self.yieldFunction(sigTrial)
                    iteration += 1
                if yieldValue < -self.yieldTolerance:
                    epsPlastic = epsPlastic - np.sqrt(deps_plastic.T @ deps_plastic)[0, 0]
                    self.hardening = self.getHardening(epsPlastic)
                    sigTrial = sigTrial + np.dot(self.D, deps_plastic).reshape(-1)
                    self.vonMises = self.getVonMises(sigTrial)
                    yieldValue = self.yieldFunction(sigTrial)
                    r_min, r_max = 0, 1.
                    while yieldValue < -self.yieldTolerance or yieldValue > 0:
                        r_mid = (r_min + r_max) / 2.
                        deps_plastic = r_mid * dLambda * dfds_mat
                        # deps_plastic = r_mid * deps_plastic
                        epsPlastic_mid = epsPlastic + np.sqrt(deps_plastic.T @ deps_plastic)[0, 0]
                        hardening = self.getHardening(epsPlastic_mid)
                        sigTrial_mid = sigTrial - np.dot(self.D, deps_plastic).reshape(-1)
                        self.vonMises = self.getVonMises(sigTrial_mid)
                        yieldValue = self.yieldFunction(sigTrial_mid, hardening=hardening)
                        if yieldValue > 0:
                            r_min = r_mid
                        else:
                            r_max = r_mid
                    self.sigTrial = sigTrial_mid
                    epsPlastic = epsPlastic_mid
                    self.hardening = hardening
                self.epsPlasticVector += deps_plastic.reshape(-1)
                self.epsPlastic = epsPlastic

                self.sig = self.sigTrial

            self.eps = self.eps + deps
            self.yieldValue = yieldValue
            self.lastYield = yieldValue
            self.loadHistoryList.append(np.array(list(self.sig) + list(self.eps) +
                                                 [self.vonMises, self.epsPlastic, self.hardening] +
                                                 list(self.epsPlasticVector) + [self.yieldValue, iteration]))
        figTitle = 'Mises_%d_%s' % (self.iterationNum, self.loadMode+str(sampleIndex) if 'random' in self.loadMode else self.loadMode)
        if 'random' in self.loadMode:
            savePath = 'MCCData'
            figTitle = os.path.join('results', figTitle)
            writeDownPaths(path='./MCCData/results', data=np.array(self.loadHistoryList), sampleIndex=sampleIndex)
        else:
            savePath = 'figSav'
            figTitle = os.path.join('MisesBaseline', figTitle)
        plotHistory(loadHistory=self.loadHistoryList,
                    figTitle=figTitle, savePath=savePath)
        # plotConfiguration2D(epsList=np.array(self.loadHistoryList)[:, 3:6], scaleFactor=75, sampleIndex=sampleIndex)

    def yieldFunction(self, sig, hardening=None):
        if hardening:
            yieldValue = self.getVonMises(sig) - hardening - self.yieldStress
        else:
            yieldValue = self.getVonMises(sig) - self.hardening - self.yieldStress
        return yieldValue

    def getHardening(self, epsPlastic):
        hardingValue = self.A * (self.epsilon0 + abs(epsPlastic)) ** self.n
        return hardingValue

    def getVonMises(self, sig):
        vonMises = np.sqrt(sig[0] ** 2 - sig[0] * sig[1] + sig[1] ** 2 + 3. * sig[2] ** 2)
        # vonMises = np.sqrt((sig[0] - sig[1])**2 + 4. * sig[2] ** 2)
        return vonMises

    def tangentAssemble(self, lam, G):
        D = np.zeros([3, 3])
        for i in range(2):
            for j in range(2):
                D[i, j] += lam
        D[0, 0] += 2 * G
        D[1, 1] += 2 * G
        D[2, 2] += G
        return D

    def getAxialDeps(self):
        dEps = np.array(
            [self.depsAxial, -self.D[1, 0] / self.D[1, 1] * self.depsAxial, 0.])
        return dEps

    def getDiffVectorOfYieldFunction(self, sig, mises, epsPlastic):
        if mises == 0:
            dfds = np.array([1, 1, np.sqrt(3)])
            # dfds = np.array([1, 1, 2])
        else:
            dfds = np.array([(2 * sig[0] - sig[1]) / 2 / mises,
                             (2 * sig[1] - sig[0]) / 2 / mises,
                             3 * sig[2] / mises])
            # dfds = np.array([(sig[0] - sig[1]) / mises,
            #                  (sig[1] - sig[0]) / mises,
            #                  4 * sig[2] / mises])
        dfdEps_p = -self.A * self.n * (self.epsilon0 + abs(self.epsPlastic)) ** (self.n - 1)
        return dfds, dfdEps_p

    def transiformationSplit(self, deps):
        """
                Used to search the point where the loading
                transform into the plasticity from the ela-
                sticity.

        :return:
        """
        r_min, r_max = 1e-64, 1.0
        r_mid = 0.5 * (r_min + r_max)
        yield_mid = self.yieldFunction(self.sig + np.dot(self.D, r_mid * deps))
        while yield_mid < -self.yieldTolerance or yield_mid > 0.:
            yield_mid = self.yieldFunction(self.sig + np.dot(self.D, r_mid * deps))
            if yield_mid < 0:
                r_min = r_mid
            else:
                r_max = r_mid
            r_mid = 0.5 * (r_min + r_max)

        return r_mid, yield_mid


def plotHistory(loadHistory, dim=2, vectorLen=3, figTitle=None, savePath='./figSav'):
    load_history = np.array(loadHistory)
    sig = load_history[..., :vectorLen]
    eps = load_history[..., vectorLen:vectorLen * 2]
    epsPlasticVector = load_history[..., (vectorLen * 2 + 3):(vectorLen * 2 + 6)]
    misesStress = load_history[..., vectorLen * 2]
    strainPlastic = load_history[..., vectorLen * 2 + 1]
    hardening = load_history[..., vectorLen * 2 + 2]
    yieldVlue = load_history[..., (vectorLen * 3 + 3):(vectorLen * 3 + 4)]
    iteration = load_history[..., (vectorLen * 3 + 4):(vectorLen * 3 + 5)]
    epsPlastic = load_history[..., (vectorLen*2+1):(vectorLen*2+2)]

    plt.figure(figsize=(16, 7))
    # strain
    ax = plt.subplot(221)
    epsLabel = ['$\epsilon_{xx}$', '$\epsilon_{yy}$', '$\epsilon_{xy}$'] if dim == 2 else \
        ['$\epsilon_{xx}$', '$\epsilon_{yy}$', '$\epsilon_{zz}$', '$\epsilon_{xy}$', '$\epsilon_{yz}$',
         '$\epsilon_{xz}$']
    plotSubFigures(ax, x=[range(len(eps)) for _ in range(len(eps[0]))], y=eps.T,
                   label=epsLabel,
                   xlabel='Load step', ylabel='$\epsilon$', num=vectorLen)

    # yield Value
    ax = plt.subplot(222)
    plotSubFigures(ax=ax, x=range(len(sig)), y=yieldVlue, label='yieldValue', xlabel='Load step', ylabel='yieldValue')
    ax2 = ax.twinx()
    ax2.plot(range(len(sig)), iteration, label='iterationNum', color='r', marker='o', lw=3)
    plt.ylabel('iterationNum', fontsize=12)
    plt.ylim([-0.5, 2.0])
    plt.legend(fontsize=15)
    plt.yticks(fontsize=12)

    # stress
    ax = plt.subplot(223)
    sigLabel = ['$\sigma_{xx}$', '$\sigma_{yy}$', '$\sigma_{xy}$'] if dim == 2 else \
        ['$\sigma_{xx}$', '$\sigma_{yy}$', '$\sigma_{zz}$', '$\sigma_{xy}$', '$\sigma_{yz}$',
         '$\sigma_{xz}$']
    plotSubFigures(ax, x=[range(len(sig)) for _ in range(len(sig[0]))], y=sig.T,
                   label=sigLabel,
                   xlabel='Load step', ylabel='$Pa$', num=vectorLen)

    #
    ax = plt.subplot(224)
    epsLabelPlastic = ['$\epsilon_{xx}^p$', '$\epsilon_{yy}^p$', '$\epsilon_{xy}^p$'] if dim == 2 else \
        ['$\epsilon_{xx}$', '$\epsilon_{yy}$', '$\epsilon_{zz}$', '$\epsilon_{xy}$', '$\epsilon_{yz}$',
         '$\epsilon_{xz}$']
    plotSubFigures(ax, x=[range(len(epsPlasticVector)) for _ in range(len(epsPlasticVector[0]))], y=epsPlasticVector.T,
                   label=epsLabelPlastic,
                   xlabel='Load step', ylabel='$\epsilon$', num=vectorLen)
    ax2 = ax.twinx()
    plotSubFigures(ax=ax2, x=range(len(sig)), y=epsPlastic, label=r'$\int |\mathrm{d}\bar{\epsilon}^p|$',
                   xlabel='Load step', ylabel='$\epsilon$', color='r')

    plt.tight_layout()
    plt.savefig('./%s/%s.png' % (savePath, figTitle if figTitle else 'Mises'), dpi=200)
    plt.close()
    return


def writeDownPaths(path, sampleIndex, data):
    """
    np.array(list(self.sig) + list(self.eps) +
                                                 [self.vonMises, self.epsPlastic, self.hardening] +
                                                 list(self.epsPlasticVector) + [self.yieldValue, iteration])
    :param path:
    :param sampleIndex:
    :param data:
    :return:
    """
    filePath = os.path.join(path, 'random_%d.dat' % sampleIndex)
    np.savetxt(fname=filePath, X=data, fmt='%10.5f', delimiter=',',
               header='sigma_xx, sigma_yy, sigma_xy, epsilon_xx, epsilon_yy, epsilon_xy, ' +
                      'vonMises, epsPlastic, hardening, ' +
                      'epsilonP__xx, epsilonP__yy, epsilonP__xy, yieldValue, iteration')


# --------------------------------------------
# main
# load path reader
if __name__ == '__main__':
    baselineFlag = True
    if not baselineFlag:
        # ----------------------------------------
        # training data generation
        loadPathList = loadingPathReader()[:2]
        print()
        print('=' * 80)
        print('\t Path loading ...')
        for i in range(len(loadPathList)):
            print('\t\tPath %d' % i)
            mises = MisesAssociateFlowIsoHarden(loadMode='random')
            mises.forward(path=loadPathList[i], sampleIndex=i)
    else:
        # ----------------------------------------
        # training data generation
        mises = MisesAssociateFlowIsoHarden(loadMode='axial')
        mises.forward()
