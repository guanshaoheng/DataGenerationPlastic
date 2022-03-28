import copy
import os
import numpy as np
import matplotlib.pyplot as plt
import sympy
from sciptes4figures.plotConfiguration2D import plotConfiguration2D
from FEMxEPxML.misesTraining import Restore, Net, DenseResNet
from MCCUtil import getQ, getP, getS, get_dpdsig_dqdsigma, getJ2, getPrinciple, getQEps, getJ2Eps, getEpsDevitoric, \
    getVolStrain, \
    getInvariantsSigma, get_b, getLode

'''
    Author: Shaoheng Gun
            Wuhan University & Swansea University
    Email: shaohengguan@gmail.com
    
    Critical state theory involved Unified Hardening model for both sand and clay

        in tensor notion

        Reference:
        [1] Yao, Y. P., Liu, L., Luo, T., Tian, Y., & Zhang, J. M. (2019).
            Unified hardening (UH) model for clays and sands. Computers and Geotechnics,
            110(March), 326–343. https://doi.org/10.1016/j.compgeo.2019.02.024

'''


class CSUH:
    def __init__(self, Z=None, e_c0=0.934, p0=1e3, e0=0.833):
        # -----------------Parameters (fundamental)------------------
        self.M = 1.25  # ratio at critical state
        self.lambdaa = 0.135
        self.kappa = 0.04
        self.poisson = 0.3
        self.N = 1.973  # location in normal consolidation on e-lnp space
        self.chi = 0.4
        self.m = 1.8
        # self.pa = 1e5  # standard pressure
        self.e0 = e0  # intial void ratio
        self.p0 = p0
        self.c_p = (self.lambdaa - self.kappa) / (1. + self.e0)
        ''' Eq. (45) Z or e_c0, only one is needed!
                
                # TODO
                e_c0 is the void ratio at \eta = 0  p = 0
                Z    is the void ratio at \eta = 0. p = 1kPa(in the paper)  while p = 1 Pa (in this code)
        '''
        if e_c0 is not None:
            self.e_c0 = e_c0  # void ratio on the critical state line at the mean effective stress p=0 kPa
            ''' Eq. (47)'''
            self.ps = np.exp((self.N - self.e_c0) / self.lambdaa)  # compressive hardening parameter corresponding to Z
            # self.Z = self.e_c0 - self.lambdaa * np.log((1+self.ps) / self.ps)
            self.Z = self.N - self.lambdaa * np.log(self.ps + 1.)
        elif Z is not None:
            self.Z = Z
            self.e_c0 = self.N - np.log(np.exp((self.N - self.Z) / self.lambdaa) - 1.) * self.lambdaa
            ''' Eq. (47)'''
            self.ps = np.exp((self.N - self.e_c0) / self.lambdaa)  # compressive hardening parameter corresponding to Z
        else:
            raise ValueError('According to Eq. (45), Z or e_c0, only one is needed!')

        # -----------------States (calculated)------------------
        # according to the current stress and void ratio state
        self.q, self.p = 0., self.p0
        self.K, self.G, self.lam = self.getElasticModulus(self.p)
        self.D = self.getMaterialMatrix(lam=self.lam, G=self.G)
        self.sigma = np.zeros([3, 3]) + np.diag([self.p, self.p, self.p])

        self.sigma_principal = np.sort(getPrinciple(sigma=self.sigma))
        self.b = get_b(*self.sigma_principal)
        self.lode = getLode(b=self.b)
        self.InvariantsSigma = getInvariantsSigma(sigma=self.sigma)
        self.e = self.e0
        self.eta = self.q / self.p
        self.e_eta = self.get_e_eta(eta=self.eta, p=self.p)
        self.xi = self.e_eta - self.e
        self.over_overconsolidation_ratio = np.exp(-self.xi / (self.lambdaa - self.kappa))
        self.M_c = self.getM_c(xi=self.xi)
        self.M_f = self.getM_f(xi=self.xi)

        # -----------------Reference yield surface (calculated)------------------
        self.px0 = 1.
        self.epsvp = 0.

        # -----------------States (Transformed Space, TS)------------------
        self.p, self.q = getP(self.sigma), getQ(sigma=self.sigma)
        self.q_ts = self.get_q_c(*getInvariantsSigma(sigma=self.sigma))
        self.sigma_ts = self.get_sigma_ts(sigma=self.sigma, q_ts=self.q_ts, q=self.q, p=self.p)
        self.eta_ts = self.q_ts / self.p
        self.e_eta_ts = self.get_e_eta(eta=self.eta_ts, p=self.p)
        self.xi_ts = self.e_eta_ts - self.e
        self.px0 = self.px0
        # self.px_ts = self.get_px(px0=self.px0, xi=self.xi_ts)
        self.px_ts = self.px0
        self.M_f_ts = self.getM_f(xi=self.xi_ts)
        self.M_c_ts = self.getM_c(xi=self.xi_ts)
        self.i1, self.i2, self.i3 = sympy.symbols('i1, i2, i3')
        self.s1, self.s2, self.s3, self.s4, self.s5, self.s6 = sympy.symbols('s1, s2, s3, s4, s5, s6')
        self.dqc_dI123_sym = self.get_dqc_dI123_sym()
        self.dI23_dsigma_sym = self.get_dI123_dsigma_sym()

        # -----------------Calculation parameters-----------------
        self.yieldTolerance = 0.05

        # -----------------Current yield surface (calculated)------------------
        self.H = 0.
        self.yieldValue = self.yieldFunction(q=self.q_ts, p=self.p, H=0., px0=self.px_ts)
        if self.yieldValue > self.yieldTolerance:
            temp = self.M ** 2 * (self.p / 1e3) ** 2 - self.chi * (self.q_ts / 1e3) ** 2
            self.px0 = self.px_ts = self.px0 = (1. + (self.q_ts / 1e3) ** 2. / temp) * self.p / 1e3
            self.yieldValue = self.yieldFunction(q=self.q_ts, p=self.p, H=0., px0=self.px_ts)
        if self.yieldValue > self.yieldTolerance:
            raise ValueError('The initial yield value (%.3e) > the tolerance (%.3e)' % (
            self.yieldValue, self.yieldTolerance))

        # ==================== initialization ended ======================
        print()

        # ==================== Calculation results
        self.results = [[self.p, self.q, self.e, self.H, self.epsvp, self.xi_ts, self.M_c_ts, self.M_f_ts]]

    def forward(self):
        '''
        undrained compression: 1

        The compression is positive and the extension is negative
        '''
        axialStrainObject = 0.1
        axialStrainArray = np.linspace(0., axialStrainObject, 500)
        for i in range(1, len(axialStrainArray)):
            print('\t step %d' % i)
            depsAxial = axialStrainArray[i] - axialStrainArray[i - 1]
            deps = np.diag([depsAxial, -.5 * depsAxial, -.5 * depsAxial])
            sig, e, p, q, q_ts, xi_ts, yieldValue, H, epsvp = self.solve(deps=deps)
            self.updateState(sig=sig, e=e, p=p, q=q, q_ts=q_ts, xi_ts=xi_ts, yieldValue=yieldValue, H=H, epsvp=epsvp)
            self.results.append([p, q, e, H, epsvp, xi_ts, self.getM_c(xi_ts), self.getM_f(xi_ts)])
        self.plotCurrentResults()
        self.writeDown()

    def solve(self, deps):
        '''
            The calculation is implemented in the Transformation Space (TS)
        '''
        e = self.e - (self.e0 + 1.) * getVolStrain(deps)
        sig = self.sigma + np.einsum('ijkl, kl->ij', self.D, deps)
        I123 = getInvariantsSigma(sigma=sig)
        p, q = getP(sigma=sig), getQ(sigma=sig)
        q_ts = self.get_q_c(*I123)
        sigma_ts = self.get_sigma_ts(sigma=sig, q=q, q_ts=q_ts, p=p)
        eta_ts = q_ts / p
        H = self.H
        px0 = self.px0
        px_ts = self.px_ts
        epsvp = self.epsvp
        yieldValue = self.yieldFunction(q=q_ts, p=p, H=self.H, px0=px_ts)

        # ------------ Elastic --------------
        if yieldValue < 0.:
            print('\t\t elastic')
            e_eta_ts = self.get_e_eta(eta=eta_ts, p=p)
            xi_ts = e_eta_ts - e
            px_ts = self.get_px(px0=px0, xi=xi_ts)
            return sig, e, p, q, q_ts, xi_ts, yieldValue, H, epsvp
        # ------------ Plastic while last step is elastic --------------
        elif yieldValue > 0. and self.yieldValue < -self.yieldTolerance:
            print('\t\t elastic 2 plastic')
            rmid, sig, yieldValue = self.transformSplit(deps=deps)
            K, G, lam = self.getElasticModulus(p=getP(sigma=sig))
            D = self.getMaterialMatrix(lam=lam, G=G)
            sig, q, q_ts, p, deps_p, D_ep, xi_ts, H, yieldValue, epsvp = self.returnMapping(
                deps=deps * (1. - rmid), sigLast=sig, Dlast=D, elast=self.e - getVolStrain(deps * rmid) * (
                        1. + self.e0))
        # ------------ Plastic --------------
        else:
            print('\t\t plastic')
            sig, q, q_ts, p, deps_p, D_ep, xi_ts, H, yieldValue, epsvp = self.returnMapping(
                deps=deps, sigLast=self.sigma, Dlast=self.D, elast=self.e)
        return sig, e, p, q, q_ts, xi_ts, yieldValue, H, epsvp

    def transformSplit(self, deps):
        rmin, rmax, rmid = 0., 1., 0.5
        sig = self.sigma + np.einsum('ijkl, kl->ij', self.D, deps * rmid)
        p = getP(sigma=sig)
        q_ts = self.get_q_c(*getInvariantsSigma(sigma=sig))
        yieldValue = self.yieldFunction(q=q_ts, p=p, H=self.H, px0=self.px_ts)
        while True:
            if yieldValue > 0.:
                rmax = rmid
            elif yieldValue < -self.yieldTolerance:
                rmin = rmid
            else:
                break
            rmid = .5 * (rmax + rmin)
            sig = self.sigma + np.einsum('ijkl, kl->ij', self.D, deps * rmid)
            p = getP(sigma=sig)
            q_ts = self.get_q_c(*getInvariantsSigma(sigma=sig))
            yieldValue = self.yieldFunction(q=q_ts, p=p, H=self.H, px0=self.px_ts)
        return rmid, sig, yieldValue

    def returnMapping(self, deps, sigLast, Dlast, elast):
        q_ts_last, q_last, p_last = self.get_q_c(*getInvariantsSigma(sigma=sigLast)), getQ(sigLast), getP(sigLast)
        sigma_ts_last = self.get_sigma_ts(sigma=sigLast, q=q_last, p=p_last, q_ts=q_ts_last)
        eta_ts_last = q_ts_last / p_last
        xi_ts_last = self.get_e_eta(eta=eta_ts_last, p=p_last) - elast
        # px0_last = self.px0
        # px = self.get_px(px0=px0_last, xi=xi_ts_last)
        Mc_last, Mf_last = self.getM_c(xi_ts_last), self.getM_f(xi_ts_last)
        yieldValue_last = self.yieldFunction(q=q_ts_last, p=p_last, H=self.H, px0=self.px_ts)
        sig = sigLast + np.einsum('ijkl, kl->ij', Dlast, deps)
        p, q = getP(sigma=sig), getQ(sigma=sig)
        # q_ts = self.get_q_c(*getInvariantsSigma(sigma=sig))
        # yieldValue = self.yieldFunction(q=q_ts, p=p, H=self.H, px0=self.px0)

        # return mapping
        dg_dsigma = self.get_dg_dsigma(Mc=Mc_last, eta=eta_ts_last, p=p_last, sigma=sigma_ts_last)
        df_dsigma, df_depsvp = self.get_df_dsigma_df_depsp(
            Mf=Mf_last, Mc=Mc_last,
            sigma_ts=sigma_ts_last, sigma=sigLast)
        temp = np.einsum('ij, ijkl, kl->', df_dsigma, Dlast, dg_dsigma) - df_depsvp * np.trace(dg_dsigma)
        A_ts = (np.einsum('ij, ijkl, kl->', df_dsigma, Dlast, deps) + yieldValue_last) / temp
        # A_ts = np.einsum('ij, ijkl, kl->', df_dsigma, Dlast, deps) / temp
        deps_p = A_ts * dg_dsigma
        depsvp = getVolStrain(deps_p)
        # if depsvp > 0.:
        #     self.plotCurrentResults()
        #     raise ValueError('The plastic strain increment is in the wrong direction (depsvp=%.3e)' % depsvp)
        D_ep = Dlast - np.einsum('ijmn, mn, st, stkl', Dlast, dg_dsigma, df_dsigma, Dlast) / temp
        # sig = self.sigma+np.einsum('ijkl, kl', D_ep, deps)
        sig = sigLast + np.einsum('ijkl, kl', Dlast, deps - deps_p)
        p = getP(sigma=sig)
        q_ts = self.get_q_c(*getInvariantsSigma(sigma=sig))
        # calculate the updated state variables
        e = elast - (1. + self.e0) * getVolStrain(deps)
        eta_ts = q_ts / p
        xi_ts = self.get_e_eta(eta=eta_ts, p=p) - e
        epsvp = self.epsvp + depsvp
        H = self.H + self.get_dH(
            Mf=0.5*(Mf_last+self.getM_f(xi_ts)),
            Mc=0.5*(Mc_last+self.getM_c(xi_ts)),
            eta=.5*(eta_ts_last+q_ts/p), depsvp=depsvp)
        yieldValue = self.yieldFunction(q=q_ts, p=p, H=H, px0=self.px0)
        print('\t\t\t Yield value: %.3e' % yieldValue)
        if np.abs(yieldValue) > self.yieldTolerance:
            self.plotCurrentResults()
            raise ValueError('The Yield Value is still in wrong value (yieldValue=%.3e)>(tolerance=%.3e)' %
                             (yieldValue, self.yieldTolerance))

        # px0 = px0_last*np.exp(-depsvp/self.c_p)
        # px = self.get_px(px0=px0, xi=xi_ts)
        return sig, q, q_ts, p, deps_p, D_ep, xi_ts, H, yieldValue, epsvp

    def updateState(self, sig, e, p, q, q_ts, xi_ts, yieldValue, H, epsvp):
        # sig = sig, e = e, p = p, q = q, q_ts = q_ts, xi_ts = xi_ts, yieldValue = yieldValue
        # self.sigma_ts = sigma_ts
        # self.eta_ts = eta_ts
        # self.e_eta_ts = e_eta_ts
        self.sigma = sig
        self.e = e
        self.p = p
        self.q = q
        self.q_ts = q_ts
        self.xi_ts = xi_ts
        self.yieldValue = yieldValue
        self.H = H
        self.epsvp = epsvp
        self.K, self.G, self.lam = self.getElasticModulus(p=p)
        self.D = self.getMaterialMatrix(lam=self.lam, G=self.G)
        self.M_c_ts = self.getM_c(xi_ts)
        self.M_f_ts = self.getM_f(xi_ts)

    def plotCurrentResults(self):
        '''
                  0  1  2  3    4      5            6                   7
                 [p, q, e, H, epsvp, xi_ts, self.getM_c(xi_ts), self.getM_f(xi_ts)]
        '''
        results = np.array(self.results[:-1])
        length = len(results)
        p, q, e, H, epsvp = results[:, 0]/1e6, results[:, 1]/1e6, results[:, 2], results[:, 3], results[:, 4]
        xi_ts, M_c, M_f = results[:, 5], results[:, 6], results[:, 7]
        eta = q / p

        fig = plt.figure(figsize=[12, 6])
        plt.xticks([])
        plt.yticks([])
        plt.title('$e_{0}$=%.3f p0=%.3f kPa' % (self.e0, self.p0 / 1e3))

        ax = fig.add_subplot(241)
        plt.plot(p, q, label='q-p')
        plt.xlabel('p MPa')
        plt.ylabel('q MPa')
        plt.axis('equal')
        plt.tight_layout()
        plt.legend()

        ax = fig.add_subplot(242)
        plt.plot(range(length), H, label='H')
        plt.tight_layout()
        plt.legend()

        ax = fig.add_subplot(243)
        plt.plot(range(length), epsvp, label='$\epsilon_v^p$')
        plt.tight_layout()
        plt.legend()

        ax = fig.add_subplot(244)
        plt.plot(range(length), q, label='$q$')
        plt.tight_layout()
        plt.legend()

        ax = fig.add_subplot(245)
        plt.plot(range(length), xi_ts, label=r'$\xi_{ts}$')
        plt.tight_layout()
        plt.legend()

        ax = fig.add_subplot(246)
        plt.plot(range(length), M_c, label='$M_{c}$')
        plt.tight_layout()
        plt.legend()

        ax = fig.add_subplot(247)
        plt.plot(range(length), eta, label='$\eta$')
        plt.tight_layout()
        plt.legend()

        ax = fig.add_subplot(248)
        plt.plot(eta, M_c, label=r'$M_{c}-\eta$')
        plt.plot(eta, eta, 'r-.')
        plt.axis('equal')
        plt.tight_layout()
        plt.legend()

        fig_name = os.path.join('CSUHresults', 'Toyoura_e0_%.3f_p0_%.3fkPa.png' % (self.e0, self.p0 / 1e3))
        plt.savefig(fig_name)
        plt.close()
        print()
        print('Fig saved as %s' % fig_name)
        return

    def writeDown(self):
        txt_name = os.path.join('CSUHresults', 'Toyoura_e0_%.3f_p0_%.3fkPa.dat' % (self.e0, self.p0 / 1e3))
        np.savetxt(fname=txt_name, X=np.array(self.results))

    def get_sigma_ts(self, sigma, q, q_ts, p):
        v = p * np.eye(3) + q_ts / q * (
            sigma - p * np.eye(3)) if q != 0. else sigma
        return v

    def get_e_eta(self, eta, p):
        ''' UH model '''
        # e_eta = self.N-self.lambdaa*np.log(self.p)-(self.lambdaa-self.kappa)*np.log(1.+self.eta**2./self.M**2.)
        ''' CSUH model'''
        p = p / 1e3
        e_eta = self.Z - self.lambdaa * np.log((p + self.ps) / (1. + self.ps)) - \
                (self.lambdaa - self.kappa) * np.log(
            ((self.M ** 2 + eta ** 2) / (self.M ** 2 - self.chi * eta ** 2) * p + self.ps) / (p + self.ps))
        return e_eta

    def getM_c(self, xi):
        ''' Eq. (33) '''
        return self.M * np.exp(-self.m * xi)

    def getM_f(self, xi):
        ''' Eq. (12) '''
        return 6. / \
               (np.sqrt(12. * (3. - self.M) / self.M ** 2 * np.exp(-xi / (self.lambdaa - self.kappa)) + 1.) + 1.)

    def get_px0(self, epsvp):
        px0 = self.px0 * np.exp(epsvp / self.c_p)
        return px0 * 1e3

    def get_px(self, px0, xi):
        px0 = px0 / 1e3
        ''' Eq. (26)'''
        px = (self.ps + px0) * np.exp(xi / self.c_p) - self.ps
        return px * 1e3

    def get_dH(self, Mf, Mc, eta, depsvp):
        return (Mf ** 4 - eta ** 4) / (Mc ** 4 - eta ** 4) * depsvp

    def NCL(self, p):
        p = p / 1e3
        ''' Eq. (20) '''
        return self.N - self.lambdaa * np.log((p + self.ps) / (1. + self.ps))

    def yieldFunction(self, q, p, H, px0, tsFlag=True):
        p /= 1e3
        q /= 1e3
        temp = self.M ** 2 * p ** 2 - self.chi * q ** 2
        if temp < 0.:
            f = 1e32
            return f
        if tsFlag:
            f = np.log((1. + q ** 2. / temp) * p + self.ps) - \
                np.log(px0 + self.ps) - \
                H / self.c_p
        else:
            f = np.log((1. + (1 + self.chi) * q ** 2. / (self.M ** 2 * p ** 2 - self.chi * q ** 2)) * p + self.ps) - \
                np.log(px0 + self.ps) - \
                H / self.c_p
        return f

    def getElasticModulus(self, p):
        K = (1. + self.e0) / self.kappa * (p/1e3 + self.ps)*1e3
        G = 3. * (1 - 2 * self.poisson) * K / 2. / (1. + self.poisson)
        lam = K - 2. / 3. * G
        return K, G, lam

    def getMaterialMatrix(self, lam, G):
        matrix = np.zeros(shape=[3, 3, 3, 3])
        for i in range(3):
            for j in range(3):
                matrix[i, i, j, j] += lam
        for i in range(3):
            matrix[i, i, i, i] += 2. * G
            matrix[i, (i + 1) % 3, i, (i + 1) % 3] = \
                matrix[i, (i + 1) % 3, (i + 1) % 3, i] = \
                matrix[(i + 1) % 3, i, (i + 1) % 3, i] = \
                matrix[(i + 1) % 3, i, i, (i + 1) % 3] = G
        return matrix

    def get_q_c(self, I1, I2, I3):
        ''' Reference eq. (50) in paper Unified hardening (UH) model for clays and sands '''
        q_c = 2. * I1 / (3. * np.sqrt((I1 * I2 - I3) / (I1 * I2 - 9. * I3)) - 1.) if I1 * I2 - 9. * I3 > 0. else 0.
        return q_c

    def get_dg_dsigma(self, Mc, eta, p, sigma):
        term1 = (Mc ** 2. - eta ** 2.) / (Mc ** 2. + eta ** 2.) * np.eye(3) / 3.
        term2 = 3. * (sigma - p * np.eye(3)) / p / (Mc ** 2. + eta ** 2.)
        v = (term1 + term2) / p
        return v

    def get_df_dsigma_df_depsp(self, Mf, Mc, sigma_ts, sigma):
        '''  sigma = np.random.random(size=[3, 3])
             sigma = 0.5*(sigma+sigma.T)
        '''
        p, q = getP(sigma_ts), getQ(sigma_ts)
        eta = q / p
        dfdp_up = self.M ** 4 - (1. + 3. * self.chi) * self.M ** 2. * eta ** 2. - self.chi * eta ** 4.
        dfdp_low = p * (self.M ** 2. - self.chi * eta ** 2.) * \
                   (self.M ** 2. + eta ** 2. + (self.M ** 2. - self.chi * eta ** 2.) * self.ps / p)
        dfdq_up = 2. * self.M ** 2. * (1 + self.chi) * eta
        dfdp = dfdp_up / dfdp_low
        dfdq = dfdq_up / dfdp_low
        dp_dsigma = np.eye(3) / 3.
        invariantsSigma = getInvariantsSigma(sigma=sigma)
        dqc_dI123 = self.get_dqc_dI(*invariantsSigma)
        dI123_dsigma = self.get_dI123_dsigma(sigma=sigma)
        dqc_dsigma = np.einsum('j, jkl->kl', dqc_dI123, dI123_dsigma)
        dfdsigma = dfdp * dp_dsigma + dfdq * dqc_dsigma
        df_depsvp = -(Mf ** 4 - eta ** 4) / (Mc ** 4 - eta ** 4) / self.c_p
        return dfdsigma, df_depsvp

    def get_dqc_dI(self, I1, I2, I3):
        if I1 * I2 - 9 * I3 < 0.:
            dqc_dI123 = np.array([3.536e5, 3.53647807e1, -1.06094342e-2])
        else:
            dqc_dI123 = np.array([self.dqc_dI123_sym[i].subs(
                {self.i1: I1, self.i2: I2, self.i3: I3}) for i in range(3)], dtype=float)
        return dqc_dI123

    def get_dI123_dsigma(self, sigma):
        '''
            https://en.wikipedia.org/wiki/Cauchy_stress_tensor

            sigma: in tensor notion
            sigma_voigt: in Voigt notion
        '''
        dI123_dsigma = [np.eye(3)] + [np.array(
            self.dI23_dsigma_sym[i].subs(
                {self.s1: sigma[0, 0], self.s2: sigma[1, 1], self.s3: sigma[2, 2],
                 self.s4: sigma[0, 1], self.s5: sigma[1, 2], self.s6: sigma[0, 2]}), dtype=float)
            for i in range(2)]
        return np.array(dI123_dsigma)

    def get_dqc_dI123_sym(self):
        '''
            Eq. (50)
        '''
        qc = 2 * self.i1 / (3. * sympy.sqrt((self.i1 * self.i2 - self.i3) / (self.i1 * self.i2 - 9 * self.i3)) - 1)
        dqc_di1 = sympy.diff(qc, self.i1)
        dqc_di2 = sympy.diff(qc, self.i2)
        dqc_di3 = sympy.diff(qc, self.i3)
        return dqc_di1, dqc_di2, dqc_di3

    def get_dI123_dsigma_sym(self):
        sigma = sympy.Matrix([[self.s1, self.s4, self.s6],
                              [self.s4, self.s2, self.s5],
                              [self.s6, self.s5, self.s3]])
        ii2 = 0.5 * (sympy.trace(sigma) ** 2 - sympy.trace(sigma ** 2))
        ii3 = sympy.det(sigma)
        di2_dsigma_sym = sympy.diff(ii2, sigma)
        di3_dsigma_sym = sympy.diff(ii3, sigma)
        return di2_dsigma_sym, di3_dsigma_sym


if __name__ == '__main__':
    # comparison of the void ratio
    # for e0 in np.linspace(0.7, 0.935, 6):
    #     csuh = CSUH(e0=e0, p0=1e6)
    #     csuh.forward()

    # comparison of the initial pressure
    for p0 in [0.1 * 1e6, 1.0 * 1e6, 2.0 * 1e6, 3.0 * 1e6]:
        csuh = CSUH(p0=p0)
        csuh.forward()
