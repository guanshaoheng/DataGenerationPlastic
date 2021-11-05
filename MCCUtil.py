import os

import numpy as np
from matplotlib import pyplot as plt


class ModifiedCamClay:
    def __init__(self, N, lambda_e, pc, kappa_e, p0, poisn, drainedFlag, M):
        """

        :param N: the y coordinate of the natural consolidation line (e-lnp)
        :param l: the slope of the e-lnp line
        :param pc_origin:
        :param k:
        :param p0: consolidation mean stress
        :param poisn:
        :param drainedFlag: drained or not
        :param M: the q/p ratio at the critical state
        :return:
        """
        # ----------------------------------------------------------------
        # material parameters definition
        self.N, self.lambda_e, self.pc, self.kappa_e, self.p = N, lambda_e, pc, kappa_e, p0
        self.poisn, self.drainedFlag, self.M = poisn, drainedFlag, M

        # ----------------------------------------------------------------
        # state variables which will be updated during the computation
        self.sig = np.array([p0] * 3 + [0.] * 3)  # The stress at current stage
        self.sig_ = np.array([p0] * 3 + [0.] * 3)  # Trial stress
        self.eps = np.zeros(6)
        self.dsig = np.zeros(6)
        self.q, self.eta = 0., 0.
        self.epsVol, self.epsDev = self.getDevVolStrain(self.eps)
        self.vol = self.N - self.lambda_e * np.log(self.pc) + self.kappa_e * np.log(self.pc / p0)
        self.vol_initial = self.vol
        self.e = self.vol - 1.
        self.dFdS = np.zeros(6)  # The vector is differential yield function to stress
        self.dFdEps_p = np.zeros(6)  # The vector is differential yield function to the plastic strain (in addition
        # that the accumulated plastic strain is chosen as the internal variable)

        # ----------------------------------------------------------------
        # loading configuration
        self.axilStrainObject = 0.2
        self.depsAxial = 0.0001
        self.interNum = int(self.axilStrainObject / self.depsAxial)

        # ----------------------------------------------------------------
        # output list
        self.loadHistoryList = [np.array(list(self.sig) + list(self.eps) + [self.e, self.epsVol, self.epsDev, self.pc])]

    def forward(self, path=None):
        """

        :return:
        """
        
        for step in range(self.interNum):
            # elastic matrix construction
            K, G = self.getKandG(self.vol, self.p)
            De = self.ElasticTangentOperator(K, G)

            deps = self.getdEps(self.depsAxial, De)
            self.sig_ = np.dot(De, deps)+self.sig
            self.p, self.q, self.eta = self.getPandQ(self.sig_)

            f_yield = self.getYieldValue(self.pc)

            if f_yield < 0.:  # elastic
                self.sig = self.sig_
            else:  # plastic
                self.pc = (self.q ** 2 / self.M ** 2 + self.p ** 2) / self.p  # renew the item of pc
                # Calculate the derives of dfds dfdep
                self.dFdS, self.dFdEps_p = self.getDiffVectorOfYieldFunction(self.pc)
                dfds_mat = self.dFdS.reshape([6, 1])
                dfdep_mat = self.dFdEps_p.reshape([6, 1])
                D = De - (De @ dfds_mat) @ (dfds_mat.T @ De) / (-dfdep_mat.T @ dfds_mat + dfds_mat.T @ De @ dfds_mat)
                deps = self.getdEps(self.depsAxial, D)
                self.sig += np.dot(D, deps)

            # update
            self.eps += deps
            self.epsVol, self.epsDev = self.getDevVolStrain(self.eps)
            self.p, self.q, self.eta = self.getPandQ(self.sig)
            self.vol *= (1-sum(deps[:3]))
            self.e = self.vol-1.
            self.loadHistoryList.append(np.array(list(self.sig) + list(self.eps) + [self.e, self.epsVol, self.epsDev, self.pc]))
        self.plotHistory()

    def getDiffVectorOfYieldFunction(self, pc):
        dFdS, dFdEps = np.zeros(6), np.zeros(6)
        dFdP = 2 * self.p - pc
        dFdQ = 2 * self.q / self.M ** 2
        dPdS, dQdS = self.getDiffVectorOfsterss()
        dFdS = dFdP * dPdS + dFdQ * dQdS
        # for m in range(3):
        #     dFdS[m] = ((2 * p - pc) / 3.0 + 3 * (sig[m] - p) / M ** 2)
        # dFdEps[0:3] = ((-p) * pc * (1 + vr) / (l - k)) * (M ** 2)
        dFdEps[0:3] = (-self.p) * pc * (1 + self.e) / (self.lambda_e - self.kappa_e)
        return dFdS, dFdEps

    def getDiffVectorOfsterss(self, ):
        dPdS, dQdS = np.zeros(6), np.zeros(6)
        dPdS = np.array([1 / 3] * 3 + [0.] * 3)
        if self.q != 0:
            dQdS[0] = (2 * self.sig_[0] - self.sig_[1] - self.sig_[2]) / self.q
            dQdS[1] = (2 * self.sig_[1] - self.sig_[2] - self.sig_[0]) / self.q
            dQdS[2] = (2 * self.sig_[2] - self.sig_[0] - self.sig_[1]) / self.q
            for i in range(3, 6):
                dQdS[i] = 3 * self.sig_[i] / self.q
        return dPdS, dQdS

    def getYieldValue(self, pc):
        f_yield = self.q ** 2 / self.M ** 2 + self.p ** 2 - self.p * pc
        f_yield = min(0, f_yield)
        return f_yield

    def getdEps(self, depsAxial, De):
        if self.drainedFlag:
            # drained
            dEps = np.array(
                [depsAxial, -De[1, 0] / (De[1, 1] + De[1, 2]) * depsAxial,
                 -De[2, 0] / (De[2, 1] + De[2, 2]) * depsAxial, 0., 0., 0.])
        else:
            # undrained
            dEps = np.array([depsAxial, -0.5 * depsAxial, -0.5 * depsAxial, 0., 0., 0.])
        return dEps

    def ElasticTangentOperator(self, K, G):
        """
                Assembling the elastic tangent operator
            :param K:
            :param G:
            :return:
            """
        De = np.zeros([6, 6])
        for i in range(6):
            if i <= 2:
                De[i, i] = K + 4 / 3 * G
            else:
                De[i, i] = G
        for i in range(3):
            for j in range(3):
                if i != j:
                    De[i, j] = K - 2 / 3 * G
        return De

    def getKandG(self, vol, p):
        K = vol * p / self.kappa_e  # according to the equation $e=-\kappa*\mathrm{ln}(p)$
        G = (3 * K * (1 - 2 * self.poisn)) / (2 * (1 + self.poisn))
        return K, G

    def getPandQ(self, sig):
        """
                Compute the equivalent shear stress.
            :param sigma:
            :return:
        """
        p = np.average(sig[:3])
        q = np.sqrt(0.5 * ((sig[0] - sig[1]) ** 2.
                           + (sig[1] - sig[2]) ** 2. + (sig[0] - sig[2]) ** 2.
                           + 6 * (sig[3] ** 2. + sig[4] ** 2. + sig[5] ** 2.)))
        eta = q / p
        return p, q, eta

    def getDevVolStrain(self, eps):
        """
            \epsilon_{dev} = \sqrt{\frac{2}{3}e_{ij}e_{ij}}
            D_2 = \frac{1}{2}e_{ij}e_{ij}
        :param eps:
        :return:
        """
        epsVol = np.sum(eps[:3])
        epsDev = np.sqrt(2. / 3. * ((eps[0] - epsVol / 3.) ** 2.
                                    + (eps[1] - epsVol / 3.) ** 2. + (eps[2] - epsVol / 3.) ** 2.
                                    + 0.5 * (eps[3] ** 2. + eps[4] ** 2. + eps[5] ** 2.)))
        return epsVol, epsDev

    def plotHistory(self,):
        """
                Input data format

        :param load_history: (raws=?, columns=16): stress1-6, strain1-6, vr, epsv, epsd, pc
        :param M: q/p ratio at critical state
        :param N: y coordinate of the natural consolidate line
        :param l: slope of the e-lnp line
        :return:
        """
        load_history, M, N, l = np.array(self.loadHistoryList), self.M, self.N, self.lambda_e
        q = load_history[:, 0] - load_history[:, 1]
        p = (load_history[:, 0] + load_history[:, 1] + load_history[:, 2]) / 3.0
        pc = load_history[:, 15]
        vr = load_history[:, 12]
        vr_final = vr[-1]
        p_final = p[-1]
        axialEps = load_history[:, 6]
        epsVol = -load_history[:, 13]

        # p-epsilon_1
        plt.figure(figsize=(16, 7))
        ax = plt.subplot(241)
        plotSubFigures(ax, x=[axialEps, axialEps], y=[p, pc], label=['p', 'pc'], xlabel='epsilon_1', ylabel='p', num=2)

        # q-epsilon_1
        ax = plt.subplot(242)
        plotSubFigures(ax, x=axialEps, y=q, label='q', xlabel='epsilon_1', ylabel='q')

        # e-epsilon_1
        ax = plt.subplot(243)
        plotSubFigures(ax, x=axialEps, y=vr, label='p_consolidation (Internal variable)',
                       xlabel='epsilon_1', ylabel='p_consolidation (Internal variable)')

        # epsilon_v-epsilon_1
        ax = plt.subplot(244)
        plotSubFigures(ax, x=axialEps, y=epsVol, label='epsilon_v', xlabel='epsilon_1', ylabel='epsilon_v')

        # epsilon_q-epsilon_1
        ax = plt.subplot(245)
        plotSubFigures(ax, x=axialEps, y=load_history[:, 14], label='epsilon_q', xlabel='epsilon_1', ylabel='epsilon_q')

        # sigma_1-epsilon_1
        ax = plt.subplot(246)
        plotSubFigures(ax, x=axialEps, y=load_history[:, 0], label='sigma_1', xlabel='epsilon_1', ylabel='sigma_1')

        # sigma_3-epsilon_1
        ax = plt.subplot(247)
        plotSubFigures(ax, x=axialEps, y=load_history[:, 2], label='sigma_3', xlabel='epsilon_1', ylabel='sigma_3')

        # epsilon_q-epsilon_1
        ax = plt.subplot(248)
        plt.plot(p, q, label='q-p')
        plt.plot(np.linspace(0, 1.5 * (max(p))), M * np.linspace(0, 1.5 * (max(p))), label='p*Mf')
        p_yield_1 = np.linspace(0, pc[0])
        q_yield_1 = M * np.sqrt((p_yield_1 * pc[0] - p_yield_1 * p_yield_1))
        p_yield_2 = np.linspace(0, pc[-1])
        q_yield_2 = M * np.sqrt((p_yield_2 * pc[-1] - p_yield_2 * p_yield_2))
        plt.plot(p_yield_1, q_yield_1, label='yield surface 1')
        plt.plot(p_yield_2, q_yield_2, label='yield surface 2')
        plt.ylabel('q')
        plt.xlabel('p')
        plt.ylim([0, max(q*1.5)])
        plt.xlim([0, max(pc)])
        plt.legend()
        # plt.show()
        plt.tight_layout()
        plt.savefig('./figSav/MCCmodel-1.png')

        # csl & ncl
        fig, ax = plt.subplots()
        ax.set_xscale('log')
        p_max = 10 * max(pc)
        px = np.linspace(1, p_max)
        ax.plot(px, N - 1.0 - l * np.log(px), label='NCL')
        # Assuming that the loading must end up in a critical state
        ax.plot(px, l * np.log(p_final) + vr_final - l * np.log(px), label='CSL')
        ax.plot(p, vr, label='Load path')
        ax.scatter(p[0], vr[0], label='start')
        ax.set_xlim(1e0, p_max)
        plt.xlabel('lg(p)')
        plt.ylabel('void ratio')
        plt.legend()
        plt.tight_layout()
        plt.savefig('./figSav/MCCmodel-2.png')


def plotSubFigures(ax, x, y, label, xlabel, ylabel, num=None):
    if num and num >= 2:
        for i in range(num):
            ax.plot(x[i], y[i], label=label[i])
    else:
        ax.plot(x, y, label=label)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def loadingPathReader():
    path = 'MCCData'
    fileList = [os.path.join(path, i) for i in os.listdir(path) if '.dat' in i]
    loadPathList = []
    for i in fileList:
        pathTemp = np.loadtxt(fname=i, delimiter=',', skiprows=1)
        loadPathList.append(pathTemp)
    return loadPathList