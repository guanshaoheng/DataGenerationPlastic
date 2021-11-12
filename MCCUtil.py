import os

import numpy as np
from matplotlib import pyplot as plt


class ModifiedCamClay:
    def __init__(self, N, lambda_e, pc, kappa_e, p0, poisn, loadMode, M, dimensionNum=3):
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
        # Solution configuration
        self.dim = dimensionNum
        if self.dim == 2:  # 2D problem
            self.vectorLen = 3
        else:  # 3D problem
            self.vectorLen = 6

        # ----------------------------------------------------------------
        # material parameters definition
        self.N, self.lambda_e, self.pc, self.kappa_e, self.p = N, lambda_e, pc, kappa_e, p0
        self.poisn, self.M = poisn, M

        # ----------------------------------------------------------------
        # state variables which will be updated during the computation
        self.sig = np.array([p0] * 3 + [0.] * 3) if self.dim == 3 else np.array([p0] * 2 + [0.])
        self.sig_ = np.array([p0] * 3 + [0.] * 3) if self.dim == 3 else np.array([p0] * 2 + [0.])
        self.eps = np.zeros(self.vectorLen)
        self.dsig = np.zeros(self.vectorLen)
        self.q, self.eta = 0., 0.
        self.epsVol, self.epsDev = self.getDevVolStrain(self.eps)
        self.vol = self.N - self.lambda_e * np.log(self.pc) + self.kappa_e * np.log(self.pc / p0)
        self.vol_initial = self.vol
        self.e = self.vol - 1.
        self.dFdS = np.zeros(self.vectorLen)  # The vector is differential yield function to stress
        self.dFdEps_p = np.zeros(self.vectorLen)  # The vector is differential yield function to the plastic strain (in addition
        # that the accumulated plastic strain is chosen as the internal variable)

        # ----------------------------------------------------------------
        # loading configuration
        self.loadMode = loadMode
        self.axilStrainObject = 0.5
        self.depsAxial = 0.0001
        self.interNum = int(self.axilStrainObject / self.depsAxial)

        # ----------------------------------------------------------------
        # output list
        self.loadHistoryList = [np.array(list(self.sig) + list(self.eps) + [self.e, self.epsVol, self.epsDev, self.pc])]

    def forward(self, numIndex=None, path=None):
        """

        :return:
        """
        # if used the customized path, then redefine the load step according to the length of the load path
        if 'random' in self.loadMode:
            self.interNum = len(path)
        for step in range(self.interNum):
            # elastic matrix construction
            lam, G = self.getKandG(self.vol, self.p)
            De = self.ElasticTangentOperator(lam, G)

            # check if this is sensible in this mode, and how to end up in the predetermined strain
            if 'random' in self.loadMode:
                if step == 0:
                    deps = path[0]
                else:
                    deps = (path[step] - path[step - 1]) * 0.1
                depsAxial = deps[0]
                deps = self.getdEps(depsAxial, De)
            else:
                deps = self.getdEps(self.depsAxial, De)
            self.sig_ = np.dot(De, deps) + self.sig
            self.p, self.q, self.eta = self.getPandQ(self.sig_)

            f_yield = self.getYieldValue(self.pc)

            if f_yield < 0.:  # elastic
                self.sig = self.sig_
            iterOutYieldSurface = 1
            while f_yield > 0 and iterOutYieldSurface <= 100:  # plastic
                self.pc = np.average([max((self.q ** 2 / self.M ** 2 + self.p ** 2) / self.p, self.pc), self.pc])  # renew the item of pc
                # Calculate the derives of dfds dfdep
                self.dFdS, self.dFdEps_p = self.getDiffVectorOfYieldFunction(self.pc)
                dfds_mat = self.dFdS.reshape([self.vectorLen, 1])
                dfdep_mat = self.dFdEps_p.reshape([self.vectorLen, 1])
                De = De - (De @ dfds_mat) @ (dfds_mat.T @ De) / (-dfdep_mat.T @ dfds_mat + dfds_mat.T @ De @ dfds_mat)
                # iteration to make sure the stress state inner the yield surface
                # if 'random' in self.loadMode:
                #     self.sig_ = self.sig+np.dot(De, deps)
                #     if iterOutYieldSurface % 20 == 0 and iterOutYieldSurface>0:
                #         print('\t Step #%d Iter #%d YielfValue: %.2f' % (step, iterOutYieldSurface, f_yield))
                #     f_yield = self.getYieldValue(self.pc)
                #     iterOutYieldSurface += 1
                # else:
                #     deps = self.getdEps(self.depsAxial, De)
                #     break
                deps = self.getdEps(depsAxial, De)
                break
            self.sig += np.dot(De, deps)

            # update
            self.eps += deps
            self.epsVol, self.epsDev = self.getDevVolStrain(self.eps)
            self.p, self.q, self.eta = self.getPandQ(self.sig)
            self.vol *= (1 - sum(deps[:self.dim]))
            # self.vol = self.N - self.lambda_e * np.log(self.pc) + self.kappa_e * np.log(self.pc / self.p)
            self.e = self.vol - 1.
            self.loadHistoryList.append(
                np.array(list(self.sig) + list(self.eps) + [self.e, self.epsVol, self.epsDev, self.pc]))
        self.plotHistory(numIndex)

    def getDiffVectorOfYieldFunction(self, pc):
        dFdS, dFdEps = np.zeros(self.vectorLen), np.zeros(self.vectorLen)
        dFdP = 2 * self.p - pc
        dFdQ = 2 * self.q / self.M ** 2
        dPdS, dQdS = self.getDiffVectorOfStress()
        dFdS = dFdP * dPdS + dFdQ * dQdS
        dFdEps[0:self.dim] = (-self.p) * pc * self.vol / (self.lambda_e - self.kappa_e)
        return dFdS, dFdEps

    def getDiffVectorOfStress(self, ):
        """
                NOTE: Check if q equals 0, how to calculate the partial of q to stress.

                    In current attemptation,  $\lim_{q\rightarrow 0} \frac{\partial q }{\partial \sigma_{11}} = 1$
                                              $\lim_{q\rightarrow 0} \frac{\partial q }{\partial \sigma_{12}} = \sqrt{3}$
        :return:
        """
        dPdS, dQdS = np.zeros(self.vectorLen), np.zeros(self.vectorLen)
        if self.dim == 3:
            dPdS[0:self.dim] = 1 / 3
            if self.q != 0:
                dQdS[0] = (2 * self.sig_[0] - self.sig_[1] - self.sig_[2]) / self.q
                dQdS[1] = (2 * self.sig_[1] - self.sig_[2] - self.sig_[0]) / self.q
                dQdS[2] = (2 * self.sig_[2] - self.sig_[0] - self.sig_[1]) / self.q
                for i in range(3, 6):
                    dQdS[i] = 3 * self.sig_[i] / self.q
            else:
                dQdS = np.array([1, 1, 1, np.sqrt(3), np.sqrt(3), np.sqrt(3)])
        else:
            dPdS[0:self.dim] = 1 / 2
            if self.q != 0:
                dQdS[0] = (self.sig_[0] - self.sig_[1]) / self.q
                dQdS[1] = (self.sig_[1] - self.sig_[0]) / self.q
                dQdS[2] = 2*self.sig_[2] / self.q
            else:
                dQdS = np.array([1, 1, 2])
        return dPdS, dQdS

    def getPandQ(self, sig):
        """
                Compute the equivalent shear stress.
            :param sigma:
            :return:
        """
        if self.dim == 3:  # 3D problem
            p = (sig[..., 0] + sig[..., 1] + sig[..., 2])/3.
            q = np.sqrt(0.5 * ((sig[..., 0] - sig[..., 1]) ** 2.
                               + (sig[..., 1] - sig[..., 2]) ** 2. + (sig[..., 0] - sig[..., 2]) ** 2.
                               + 6 * (sig[..., 3] ** 2. + sig[..., 4] ** 2. + sig[..., 5] ** 2.)))
        else:  # 2D problem
            p = (sig[..., 0] + sig[..., 1])/2.
            q = np.sqrt((sig[..., 0] - sig[..., 1]) ** 2. + 4*sig[..., 2]**2)
        eta = q / p
        return p, q, eta

    def getYieldValue(self, pc):
        f_yield = self.q ** 2 / self.M ** 2 + self.p ** 2 - self.p * pc
        # f_yield = min(0, f_yield)
        return f_yield

    def getdEps(self, depsAxial, De):
        if self.loadMode == 'drained' or self.loadMode=='random':
            if self.dim == 3:
                # drained
                dEps = np.array(
                    [depsAxial, (De[1, 2] * De[2, 0] - De[2, 2] * De[1, 0]) / (
                            De[1, 1] * De[2, 2] - De[1, 2] * De[2, 1]) * depsAxial,
                     (De[2, 1] * De[1, 0] - De[1, 1] * De[2, 0]) / (
                             De[1, 1] * De[2, 2] - De[1, 2] * De[2, 1]) * depsAxial
                        , 0., 0., 0.])
            else:
                dEps = np.array(
                    [depsAxial, -De[1, 0] / De[1, 1] * depsAxial, 0.])
        else:
            if self.dim == 3:
                # undrained
                dEps = np.array([depsAxial, -0.5 * depsAxial, -0.5 * depsAxial, 0., 0., 0.])
            else:
                dEps = np.array([depsAxial, -depsAxial, 0.])
        return dEps

    def ElasticTangentOperator(self, lam, G):
        """
                Assembling the elastic tangent operator
            :param K:
            :param G:
            :return:
        """
        if self.dim == 3:
            De = np.zeros([6, 6])
            for i in range(6):
                if i <= 2:
                    De[i, i] = 2*G
                else:
                    De[i, i] = G
            for i in range(3):
                for j in range(3):
                    De[i, j] += lam
        else:
            De = np.zeros([3, 3])
            for i in range(3):
                if i < 2:
                    De[i, i] = 2*G
                else:
                    De[i, i] = G
            for i in range(2):
                for j in range(2):
                    De[i, j] += lam
        return De

    def getKandG(self, vol, p):
        """
            NOTE: K is the same under 2D and 3D conditions.

        :param vol:
        :param p:
        :return:
        """
        K = vol * p / self.kappa_e  # according to the equation $e=-\kappa*\mathrm{ln}(p)$
        if self.dim == 3:
            E = 3*K*(1-2*self.poisn)
        else:
            E = 2*K*(1-self.poisn)
        G = E / (2 * (1 + self.poisn))
        lam = E*self.poisn/(1+self.poisn)/(1-2*self.poisn)
        return lam, G

    def getDevVolStrain(self, eps):
        """
            \epsilon_{dev} = \sqrt{\frac{2}{3}e_{ij}e_{ij}}
            D_2 = \frac{1}{2}e_{ij}e_{ij}
        :param eps:
        :return:
        """
        if self.dim == 3:  # 3D problem
            epsVol = eps[..., 0] + eps[..., 1] + eps[..., 2]
            epsDev = np.sqrt(2. / 3. * ((eps[..., 0] - epsVol / 3.) ** 2.
                                        + (eps[..., 1] - epsVol / 3.) ** 2. + (eps[..., 2] - epsVol / 3.) ** 2.
                                        + 0.5 * (eps[..., 3] ** 2. + eps[..., 4] ** 2. + eps[..., 5] ** 2.)))
        else:  # 2D problem
            epsVol = eps[..., 0] + eps[..., 1]
            epsDev = np.sqrt(2. / 3. * ((eps[..., 0] - epsVol / 3.) ** 2.
                                        + (eps[..., 1] - epsVol / 3.) ** 2. + 0.5 * (eps[..., 2] ** 2.)))
        return epsVol, epsDev

    def plotHistory(self, numIndex=None):
        """
                Input data format

        :param load_history: (raws=?, columns=16): stress1-6, strain1-6, vr, epsv, epsd, pc
        :param M: q/p ratio at critical state
        :param N: y coordinate of the natural consolidate line
        :param l: slope of the e-lnp line
        :return:
        """
        load_history, M, N, l = np.array(self.loadHistoryList)[:1000, :], self.M, self.N, self.lambda_e
        sig = load_history[..., :self.vectorLen]
        eps = load_history[..., self.vectorLen:self.vectorLen*2]
        p, q, eta = self.getPandQ(sig)
        pc = load_history[..., self.vectorLen*2+3]
        vr = load_history[..., self.vectorLen*2]
        vr_final = vr[-1]
        p_final = p[-1]
        axialEps = load_history[:, self.vectorLen]
        epsVol, epsDev = self.getDevVolStrain(eps)
        epsVol = -epsVol
        eps2 = load_history[:, self.vectorLen+1]
        loadStepX = range(len(load_history))

        # p-epsilon_1
        plt.figure(figsize=(16, 7))
        ax = plt.subplot(241)
        plotSubFigures(ax, x=[loadStepX, loadStepX], y=[p, pc], label=['p', 'pc'], xlabel='Load step', ylabel='p', num=2)

        # q-epsilon_1
        ax = plt.subplot(242)
        plotSubFigures(ax, x=loadStepX, y=q, label='q', xlabel='Load step', ylabel='q')

        # e-epsilon_1
        ax = plt.subplot(243)
        plotSubFigures(ax, x=loadStepX, y=vr, label='$e$',
                       xlabel='epsilon_1', ylabel='$e$')

        # strain
        ax = plt.subplot(244)
        plotSubFigures(ax, x=loadStepX, y=load_history[:, 0], label='$\sigma_1$', xlabel='Load step', ylabel='$\sigma_1$')

        # epsilon_q-epsilon_1
        ax = plt.subplot(245)
        plotSubFigures(ax, x=loadStepX, y=epsDev, label='epsilon_q', xlabel='Load step', ylabel='epsilon_q')

        # sigma_1-epsilon_1
        ax = plt.subplot(246)
        epsLabel = ['$\epsilon_{xx}$', '$\epsilon_{yy}$', '$\epsilon_{xy}$'] if self.dim == 2 else \
            ['$\epsilon_{xx}$', '$\epsilon_{yy}$', '$\epsilon_{zz}$', '$\epsilon_{xy}$', '$\epsilon_{yz}$',
             '$\epsilon_{xz}$']
        plotSubFigures(ax, x=[range(len(eps)) for _ in range(len(eps[0]))], y=eps.T,
                       label=epsLabel,
                       xlabel='Load step', ylabel='$\epsilon$', num=self.vectorLen)

        # sigma_3-epsilon_1
        ax = plt.subplot(247)
        sigLabel = ['$\sigma_{xx}$', '$\sigma_{yy}$', '$\sigma_{xy}$'] if self.dim == 2 else \
            ['$\sigma_{xx}$', '$\sigma_{yy}$', '$\sigma_{zz}$', '$\sigma_{xy}$', '$\sigma_{yz}$',
             '$\sigma_{xz}$']
        plotSubFigures(ax, x=[range(len(sig)) for _ in range(len(sig[0]))], y=sig.T,
                       label=sigLabel,
                       xlabel='Load step', ylabel='$\sigma$', num=self.vectorLen)
        # plotSubFigures(ax, x=axialEps, y=load_history[:, 1], label='$\sigma_2$', xlabel='$\epsilon_1$', ylabel='$\sigma_3$')

        # epsilon_q-epsilon_1
        ax = plt.subplot(248)
        plt.plot(p, q, label='q-p')
        plt.plot(np.linspace(0, 1.5 * (max(p))), M * np.linspace(0, 1.5 * (max(p))), label='p*Mf')
        p_yield_1 = np.linspace(0, pc[0], 100)
        q_yield_1 = M * np.sqrt((p_yield_1 * pc[0] - p_yield_1 * p_yield_1))
        p_yield_2 = np.linspace(0, pc[-1], 100)
        q_yield_2 = M * np.sqrt((p_yield_2 * pc[-1] - p_yield_2 * p_yield_2))
        plt.plot(p_yield_1, q_yield_1, label='yield surface 1')
        plt.plot(p_yield_2, q_yield_2, label='yield surface 2')
        plt.ylabel('q')
        plt.xlabel('p')
        plt.ylim([0, max(q * 1.5)])
        plt.xlim([0, max(pc)])
        plt.legend()
        # plt.show()
        plt.tight_layout()
        if 'random' in self.loadMode:
            figName = './MCCData/MCCmodel-1_%dD_%d.png' % (self.dim, numIndex)
        else:
            figName = './figSav/MCCmodel-1_%dD.png' % self.dim
        plt.savefig(figName, dpi=200)

        if numIndex:
            pass
        else:
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
            plt.savefig('./figSav/MCCmodel-2_%dD.png' % self.dim, dpi=200)


def plotSubFigures(ax, x, y, label, xlabel, ylabel, num=None, color=None):
    if num and num >= 2:
        for i in range(num):
            ax.plot(x[i], y[i], label=label[i], lw=3)
    else:
        if color:
            ax.plot(x, y, label=label, lw=3, color=color)
        else:
            ax.plot(x, y, label=label, lw=3)
    plt.legend(fontsize=15)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)


def loadingPathReader():
    path = 'MCCData'
    fileList = [os.path.join(path, i) for i in os.listdir(path) if '.dat' in i]
    loadPathList = []
    for i in fileList:
        pathTemp = np.loadtxt(fname=i, delimiter=',', skiprows=1)
        loadPathList.append(pathTemp)
    return loadPathList
