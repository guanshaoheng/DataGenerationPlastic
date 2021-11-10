import numpy as np
import matplotlib.pyplot as plt
from MCCUtil import plotSubFigures


class MisesAssociateFlowIsoHarden:
    def __init__(self):

        #----------------------------------------------------
        # material parameters
        self.youngsModulus = 200e9
        self.poisson = 0.3
        self.A = 500e6
        self.n = 0.2
        self.epsilon0 = 0.05
        self.yieldStress = 200e6
        self.hardening = 0.
        self.D = self.tangentAssemble(lam=self.youngsModulus*self.poisson/(1+self.poisson)/(1-2*self.poisson),
                                      G=self.youngsModulus/2/(1+self.poisson))

        # ---------------------------------------------------
        # state variables [stress and strain vector in voigt notion]
        self.sig = np.zeros(3)
        self.vonMises = .0
        self.eps = np.zeros(3)
        self.epsPlastic = 0.
        self.epsPlasticVector = np.zeros(3)
        self.sigTrial = np.zeros(3)
        self.lastYield = -1
        self.loadHistoryList = [np.array(list(self.sig)+list(self.eps)+[self.vonMises, self.epsPlastic, self.hardening]+list(self.epsPlasticVector))]

        # ---------------------------------------------------
        # load configuration
        self.epsAxialObject = 0.01
        self.iterationNum = int(1e4)
        self.depsAxial = self.epsAxialObject/self.iterationNum

    def forward(self):
        for i in range(self.iterationNum):
            deps = self.getDeps(self.D, self.depsAxial)
            self.sigTrial = self.sig + np.dot(self.D, deps)
            self.vonMises = self.getVonMises(self.sigTrial)
            self.hardening = self.getHardening(self.epsPlastic)
            yieldValue = self.yieldFunction(self.sigTrial)

            if yieldValue <= 0:
                self.sig = self.sigTrial
                self.lastYield = yieldValue
                deps_plastic = np.zeros(3)
                print('elastic')
            elif abs(self.lastYield) > 1.:  # last step is elastic
                r_min, r_max = 1e-32, 1.0
                r_mid = 0.5*(r_min+r_max)
                yield_mid = self.yieldFunction(self.sig + np.dot(self.D, r_mid * deps))
                while abs(yield_mid) > 1. or yield_mid>0.:
                    yield_mid = self.yieldFunction(self.sig + np.dot(self.D, r_mid*deps))
                    if yield_mid < 0:
                        r_min = r_mid
                    else:
                        r_max = r_mid
                    r_mid = 0.5*(r_min+r_max)
                self.sig = self.sig+np.dot(self.D, r_mid * deps)
                self.vonMises = self.getVonMises(self.sig)
                self.eps += r_mid*deps
                self.loadHistoryList.append(np.array(list(self.sig)+list(self.eps)+[self.vonMises, self.epsPlastic, self.hardening]+list(self.epsPlasticVector)))
                deps = (1-r_mid)*deps
                self.lastYield = yield_mid
                # update the trial stress
                self.sigTrial = self.sig + np.dot(self.D, deps)
                self.vonMises = self.getVonMises(self.sigTrial)
                yieldValue = self.yieldFunction(self.sigTrial)

            if yieldValue <=0:
                pass
            else:               # last step is plastic
                # print('plastic')
                iteration = 0
                while yieldValue > 0 and iteration < 100:
                    self.dFdS, self.dFdEps_p = self.getDiffVectorOfYieldFunction()
                    dfds_mat = self.dFdS.reshape([3, 1])
                    h = -self.dFdEps_p*np.sqrt(2/3*(dfds_mat.T @ dfds_mat))[0, 0]
                    # Dep = self.D - (self.D @ dfds_mat) @ (dfds_mat.T @ self.D) / \
                    #       (h+ dfds_mat.T @ self.D @ dfds_mat)
                    H = (h+dfds_mat.T@self.D@dfds_mat)[0, 0]
                    dLambda = (dfds_mat.T@self.D@deps.reshape([-1, 1])/H)[0, 0]
                    deps_plastic = dLambda*dfds_mat
                    epsPlasticTrial = self.epsPlastic + np.sqrt(deps_plastic.T@deps_plastic)[0, 0]
                    self.hardening = self.getHardening(epsPlasticTrial)
                    dsig = np.dot(self.D, (deps.reshape([-1, 1])-deps_plastic))
                    self.sigTrial = self.sig+dsig.reshape(-1)
                    self.vonMises = self.getVonMises(self.sigTrial)
                    yieldValue = self.yieldFunction(self.sigTrial)
                    iteration += 1

                self.sig = self.sigTrial
            self.eps = self.eps + deps
            self.epsPlasticVector += deps_plastic.reshape(-1)
            self.loadHistoryList.append(np.array(list(self.sig)+list(self.eps)+[self.vonMises, self.epsPlastic, self.hardening]+list(self.epsPlasticVector)))
        plotHistory(loadHistory=self.loadHistoryList)

    def yieldFunction(self, sig):
        yieldValue = self.getVonMises(sig)-self.hardening-self.yieldStress
        return yieldValue

    def getHardening(self, epsPlastic):
        hardingValue = self.A*(self.epsilon0+abs(epsPlastic))**self.n
        return hardingValue

    def getVonMises(self, sig):
        vonMises = np.sqrt(sig[0] ** 2 - sig[0] * sig[1] + sig[1] ** 2 + 3. * sig[2] ** 2)
        return vonMises

    def tangentAssemble(self, lam, G):
        D = np.zeros([3, 3])
        for i in range(2):
            for j in range(2):
                D[i, j] += lam
        D[0, 0] += 2*G
        D[1, 1] += 2*G
        D[2, 2] += G
        return D

    def getDeps(self, De, depsAxial):
        dEps = np.array(
            [depsAxial, -De[1, 0] / De[1, 1] * depsAxial, 0.])
        return dEps

    def getDiffVectorOfYieldFunction(self):
        if self.vonMises == 0:
            dfds = np.array([1, 1, np.sqrt(3)])
        else:
            dfds = np.array([(2*self.sigTrial[0]-self.sigTrial[1])/2/self.vonMises,
                             (2*self.sigTrial[1]-self.sigTrial[0])/2/self.vonMises,
                             3*self.sigTrial[2]/self.vonMises])
        dfdEps_p = -self.A*self.n*(self.epsilon0+abs(self.epsPlastic))**(self.n-1)
        return dfds, dfdEps_p


def plotHistory(loadHistory, dim=2, vectorLen=3):
    load_history = np.array(loadHistory)
    sig = load_history[..., :vectorLen]
    eps = load_history[..., vectorLen:vectorLen * 2]
    epsPlastic = load_history[..., (vectorLen * 2+3):]
    misesStress = load_history[..., vectorLen * 2]
    strainPlastic = load_history[..., vectorLen * 2 + 1]
    hardening = load_history[..., vectorLen * 2 + 2]
    # vr = load_history[..., vectorLen * 2]
    # vr_final = vr[-1]
    # axialEps = load_history[:, vectorLen]
    # epsVol = np.average(sig[:, :2], axis=1)
    # eps2 = load_history[:, vectorLen + 1]
    # loadStepX = range(len(load_history))

    plt.figure(figsize=(16, 7))
    # sigma_1-epsilon_1
    ax = plt.subplot(121)
    epsLabel = ['$\epsilon_{xx}$', '$\epsilon_{yy}$', '$\epsilon_{xy}$'] if dim == 2 else \
        ['$\epsilon_{xx}$', '$\epsilon_{yy}$', '$\epsilon_{zz}$', '$\epsilon_{xy}$', '$\epsilon_{yz}$',
         '$\epsilon_{xz}$']
    plotSubFigures(ax, x=[range(len(eps)) for _ in range(len(eps[0]))], y=eps.T,
                   label=epsLabel,
                   xlabel='Load step', ylabel='$\epsilon$', num=vectorLen)
    epsLabelPlastic = ['$\epsilon_{xx}^p$', '$\epsilon_{yy}^p$', '$\epsilon_{xy}^p$'] if dim == 2 else \
        ['$\epsilon_{xx}$', '$\epsilon_{yy}$', '$\epsilon_{zz}$', '$\epsilon_{xy}$', '$\epsilon_{yz}$',
         '$\epsilon_{xz}$']
    plotSubFigures(ax, x=[range(len(epsPlastic)) for _ in range(len(epsPlastic[0]))], y=epsPlastic.T,
                   label=epsLabelPlastic,
                   xlabel='Load step', ylabel='$\epsilon$', num=vectorLen)

    # sigma_3-epsilon_1
    ax = plt.subplot(122)
    sigLabel = ['$\sigma_{xx}$', '$\sigma_{yy}$', '$\sigma_{xy}$'] if dim == 2 else \
        ['$\sigma_{xx}$', '$\sigma_{yy}$', '$\sigma_{zz}$', '$\sigma_{xy}$', '$\sigma_{yz}$',
         '$\sigma_{xz}$']
    plotSubFigures(ax, x=[range(len(sig)) for _ in range(len(sig[0]))], y=sig.T,
                   label=sigLabel,
                   xlabel='Load step', ylabel='$\sigma$', num=vectorLen)

    plt.savefig('./figSav/Mises.png', dps=200)


mises = MisesAssociateFlowIsoHarden()
mises.forward()