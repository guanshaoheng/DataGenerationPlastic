#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

"""
    Usage:
        Calculating and display the yield surface for the sand model.
        
    Reference:
        [1] Jefferies, M. G. (1993). Nor-Sand: A simple critical state model for sand. Geotechnique, 43(1), 91–103. 
            https://doi.org/10.1680/geot.1993.43.1.91
"""


class NorSand:
    def __init__(self):
        '''
            ---------------------------------------------------------------------------
                                Material constants assignment (state-independent)
             '''
        self.G_0, self.nG = 1e8, 0.1
        self.nu, self.p_ref = 0.2, 1e5
        self.e_o = 0.15
        self.K, self.G = 1e8, 1e8
        self.M = 1.25
        self.N = 0.2  # Volumetric coupling coefficient
        self.CHI = 0.5  # Dilatancy coefficient (Jefferies&Shuttle 2002) [-]
        self.psi0 = 0.2
        self.h = 100
        self.M_tc = 1.25
        self.chi_tc = 0.1
        self.OCR = 2.
        self.Gamma = 0.8
        self.lambda_c = 0.0185
        self.lambda_e = 0.0185
        self.chi_i = self.chi_tc * self.M_tc / (self.M_tc - self.lambda_c * self.chi_tc)

        '''
            ---------------------------------------------------------------------------
                            Variables related with current state
                '''
        # self.sig = np.zeros(6)
        # self.dsig = np.zeros(6)
        # self.p, self.q, self.eta = self.getPandQ(self.sig)
        # self.J2, self.J3, self.dJ2dSig, self.dJ3dSig = self.getInvariants(self.sig)
        # self.dFdSP = np.zeros(2)
        # self.eps = np.zeros(6)
        # self.epsVol, self.epsDev = self.getDevVolStrain(self.eps)
        # self.DE, self.DDSDDE = np.zeros([6, 6]), np.zeros([6, 6])
        # self.e = 0.5
        # self.eci = 0.
        # self.xM = 0.
        # self.CHIi = 0.
        # self.psi = self.e - self.eci
        # self.M_i = 1.25
        # self.p_i = 1e5
        # self.locus = False
        # self.yieldValue = 0.

        '''
            ---------------------------------------------------------------------------
                                    Tolerance
            '''
        self.FTOL = 1e-5  # Tolerance of the yield value
        self.SPTOL = 1e-5

    def mainCalculation(self, eps, deps):
        # -----------------------------------------------------------------------------
        #                   Variables related with current state definition
        sig = np.zeros(6)
        p, q, eta = self.getPandQ(sig)
        e = 0.
        G = self.G_0 * (p / self.p_ref) ** self.nG
        K = 2 * G * (1 + self.nu) / (3 * (1 - 2 * self.nu))
        e = self.e_o

        # -----------------------------------------------------------------------------
        #                          Elastic trial
        DE = self.TangentMatrixAssembling(K, G)
        dsig = DE @ deps.reshape(6, 1)
        sigNew = sig+dsig
        epsVol, epsDev = self.getDevVolStrain(eps)
        epsVol_1, epsDev_1 = self.getDevVolStrain(eps + deps)
        depsDev = epsDev_1 - epsDev
        p_i, M_i, psi = self.getP_M_Image(sigNew, e)  # calculate the p_i, M_i and \psi at the initial state
        yieldValue, locus = self.getYieldFunctionNorSand(p, q, p_i, M_i, psi)

        # -----------------------------------------------------------------------------
        #                         check plasticity
        if yieldValue < self.FTOL:  # elastic
            DDSDDE = DE


    def yitaCalculation(self, p, NN):
        if NN == 0:
            q = p * self.M * (np.log(self.p_i / p) + 1.)
        else:
            q = p * self.M / NN * (1 + 1 *
                                   ((p / self.p_i) ** (NN / (1 - NN))) * (NN - 1))
        return q

    def getP_M_Image(self, sig, e):
        """
            Function to get the p_i, M_i, and psi
        :param sig:
        :return:
        """
        # compute p, q, eta
        p, q, eta = self.getPandQ(sig)
        # Correct M due to Lode's angle
        M = self.getMlode(sig)
        # compute p_i assuming q=0
        p_i = p / np.e * self.OCR
        e_ci = self.Gamma - self.lambda_c * np.log(-p_i)  # Get critical void ratio
        psi = e - e_ci
        M_i = self.getMiwithPsi(M, psi)
        # now correct if stresses are not spherical
        if q > 0.:  # Needs to iterate e.g. in K_0 conditions
            # !I use Newton-Raphson to retrieve the initial state parameters
            pi_old = 0.
            F_pi = -1.0
            while abs(pi_old - p_i) > self.SPTOL or (F_pi < 0.):
                pi_old = p_i
                # Evaluates yield function at Sig_0 at p_i
                self.getYieldFunctionNorSand()
                # Evaluate derivative
                dFdSP = self.getdFdSP(p, p_i, psi, M, M_i)
                # Get new p_i
                p_i = p_i - (F_pi / dFdSP)
                e_ci = self.Gamma - self.lambda_e * np.log(-p_i)  # Get critical void ratio
                psi_i = e - e_ci
                M_i = self.getMiwithPsi(M, psi)
            self.p_i = self.p_i * self.OCR
        return p_i, M_i, psi

    def getYieldFunctionNorSand(self, p, q, p_i, M_i, psi):
        """
            Yield function or plastic potential surface for Nor-Sand
        :return: 
        """
        p_max = p_i / np.exp(-self.chi_i * psi / M_i)
        yieldValue= q + p * M_i * (1. + np.log(p_i / p))
        F2 = p - p_max
        locus = False
        return yieldValue, locus

    def getMlode(self, sig):
        J2, J3, _ , _ = self.getInvariants(sig)
        theta = 0.
        cos3Theta = 0.
        J3AJ3 = 0.
        sin3Theta = 0.
        if (J2 == 0.):
            J3AJ3 = 0.
        else:
            J3AJ3 = J3 / np.sqrt(J2 ** 3)

        sin3Theta = 3. * np.sqrt(3.) / 2. * J3AJ3
        sin3Theta = max(min(sin3Theta, 0.99), -0.99)
        theta = np.arcsin(sin3Theta) / 3.
        theta = max(min(sin3Theta, 0.523598), -0.523598)
        cos3Theta = np.cos(3. * theta)
        if -1e-8 < cos3Theta < 1e-8:
            cos3Theta = 0.
        M = self.M_tc - self.M_tc ** 2. / (3. + self.M_tc) * np.cos(-3. * theta / 2. + np.pi / 4.)
        return M

    def getMiwithPsi(self, M, psi):
        """
            Gets the static M_i with changes in the state parameter
	            [1] Jeffereis and Shuttle 2011
        :return:
        """
        M_i = M * (1. - (self.N * self.chi_i * abs(psi) / self.M_tc))
        return M_i

    def qMCC(self, p):
        return self.M * np.sqrt(p * (self.p_i * 2. - p))

    def TangentMatrixAssembling(self, K, G):
        D = np.zeros([6,6])
        temp1, temp2 = K + (4 * G / 3), K - (2 * G / 3)
        D[0:3, 0:3] = temp2
        D[0, 0] = temp1
        D[1, 1] = temp1
        D[2, 2] = temp1
        D[3, 3] = G
        D[4, 4] = G
        D[5, 5] = G
        return D

    def getPandQ(self, sig):
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

    def getInvariants(self, sig):
        S = self.getSigDev(sig)

        J2 = 1. / 6. * ((sig[1] - sig[2]) ** 2 +
                             (sig[2] - sig[0]) ** 2 + (sig[0] - sig[1]) ** 2) + \
                  sig[3] ** 2 + sig[4] ** 2 + sig[5] ** 2
        J3 = S[0] * S[1] * S[2] - S[0] * S[5] ** 2 - S[1] * S[4] ** 2 - S[2] * S[3] ** 2 + 2 * S[3] * S[5] * S[4]

        dJ2dSig, dJ3dSig = np.zeros(6), np.zeros(6)
        dJ2dSig[0] = S[0]
        dJ2dSig[1] = S[1]
        dJ2dSig[2] = S[2]
        dJ2dSig[3] = 2. * sig[3]
        dJ2dSig[4] = 2. * sig[4]  # In the conventional tension as positive the sig here is +
        dJ2dSig[5] = 2. * sig[5]
        dJ3dSig[0] = -1. / 3. * S[0] * S[1] - 1. / 3. * S[0] * S[2] + 2. / 3. * S[1] * S[2] - \
                          2. / 3. * S[5] ** 2 + 1. / 3. * S[4] ** 2 + 1. / 3. * S[3] ** 2
        dJ3dSig[1] = -1. / 3. * S[0] * S[1] + 2. / 3. * S[0] * S[2] - 1. / 3. * S[1] * S[2] + \
                          1. / 3. * S[5] ** 2 - 2. / 3. * S[4] ** 2 + 1. / 3. * S[3] ** 2
        dJ3dSig[2] = 2. / 3. * S[0] * S[1] - 1. / 3. * S[0] * S[2] - 1. / 3. * S[1] * S[2] + \
                          1. / 3. * S[5] ** 2 + 1. / 3. * S[4] ** 2 - 2. / 3. * S[3] ** 2
        dJ3dSig[3] = -2. * S[2] * S[3] + 2. * S[5] * S[4]
        dJ3dSig[4] = -2. * S[1] * S[4] + 2. * S[3] * S[5]
        dJ3dSig[5] = -2. * S[0] * S[5] + 2. * S[3] * S[4]
        return J2, J3, dJ2dSig, dJ3dSig

    def getSigDev(self, sig):
        p = np.average(sig[:3])
        sig[:3] = sig[:3] - p
        return sig

    def getdFdSig(self, sig, p_i, M_i, M_tc, CHIi, chi_tce, N, psi, dFdSig, dPPdSig):
        """
            Get the derivatives
            get dFdSig and dPPdSig for inner cap evaluated at Sig, M_i, p_i, psi_i
        :return:
        """
        pi = np.pi
        p, q, eta = self.getPandQ()
        # calculating dPdSig, dQdSig
        dPdSig, dQdSig = np.array([1 / 3., 1 / 3., 1 / 3., 0., 0., 0.]), np.zeros(6)
        if q != 0:
            for i in range(3):
                dQdSig[i] = 3. / 2. / q * (sig[i] - p)
            for i in range(3, 6):
                dQdSig[i] = 3. / q * (sig[i])
        #
        dJ2dSig, dJ3dSig, dThetadSig = np.zeros(6), np.zeros(6), np.zeros(6)
        self.getInvariants(sig)
        return

    def getdFdSP(self, p, p_i, psi, M, M_i):
        """

        :param p:   mean stress
        :param p_i: image mean stress
        :param psi: psi at current state
        :param M:   M at current state
        :param M_i: image M
        :param chi_tce:  current dilatancy coefficient
        :return: dFdSP:
        """
        M_itc = self.M_tc * (1 - self.chi_i * self.N * abs(-psi) / self.M_tc)
        # p_max = p_i / np.exp(-self.chi_i * psi / M_itc)
        dFdSP = np.zeros(2)
        if not self.locus:  # Call dF1 / dSP (Maybe outside the cap)
            dFdM = p * (1.0 + np.log(p_i / p))
            dFdSP[0] = -dFdM * M * self.N * self.chi_i * abs(psi) * self.lambda_c / \
                       (self.M_tc * psi * p_i) + (M_i * p / p_i)
            # Here Xhi_tc is set as the second state variable for handling the strain rates
            # dFdChitc=dFdMi*dMidChii*dChiidXhitc
            dFdSP[1] = -dFdM * M * self.N * abs(psi) / \
                       (self.M_tc * (1 - self.chi_i * self.lambda_c / self.M_tc) ** 2.0)
        else:  # Inner cap
            dpmaxdpi = np.exp(self.chi_i * psi / M_i) * (1.0 + ((self.chi_i * self.lambda_c / M_i) *
                                                                         (1.0 + (self.chi_i * self.N * abs(
                                                                             psi) / M_i))))
            dFdSP[0] = -dpmaxdpi
            dpmaxdChi_tc = p_i * psi * np.exp(self.chi_i * psi / M_i) * \
                           (1.0 + self.chi_i * self.N * abs(psi) / M_i) / \
                           (self.M_tc * (1 - self.chi_i * self.lambda_c / self.M_tc) ** 2)
            dFdSP[1] = -dpmaxdChi_tc
        return dFdSP

    def getdSpdEpsp(self, ):
        return

    def resetPi(self, p, q, NN):
        yita = q / p
        if NN != 0:
            self.p_i = p * ((1. - NN * yita / self.M) / (1 - NN)) ** (1 - 1 / NN)
        else:
            self.p_i = p * np.exp(yita / self.M - 1.)

    def hardening(self):
        return

    def qpYieldDisplay(self):
        p = np.linspace(0.1, 2 * self.p_i, 100)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(p / self.p_i, self.qMCC(p) / self.p_i / self.M, label='MCC', linewidth=3)
        for NN in [0., 0.25, 0.5]:
            if NN == 0:
                pMax = np.exp(1) * self.p_i
            else:
                pMax = (1 - NN) ** (1 - 1 / NN) * self.p_i
            p = np.linspace(0.1, pMax, 100)
            q = self.yitaCalculation(p, NN)
            ax.plot(p / self.p_i, q / self.p_i / self.M, label="N=%.2f" % NN, linewidth=3)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    yieldSurface = YieldSurface()
    yieldSurface.qpYieldDisplay()
