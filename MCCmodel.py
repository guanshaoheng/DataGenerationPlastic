import copy
import os
import numpy as np
import matplotlib.pyplot as plt
from FEMxEPxML.MCCUtil import plotSubFigures, loadingPathReader
from sciptes4figures.plotConfiguration2D import plotConfiguration2D
from FEMxEPxML.misesTraining import Restore, Net, DenseResNet

"""
        MCC model

        Author: Shaoheng Guan
        Email:  shaohengguan@gmail.com

        Reference:
            [1] http://docs.itascacg.com/3dec700/common/models/camclay/doc/modelcamclay.html
            
        *************Pending problem***************
        1. add the extension and compression check to the program
        2. solve the transforming (from elastic to plastic, the 
            M_compression changes to M_extension, so bug arise in 
            the the transformSplit function ) problem as Path 5, 
            iteration at 859-860.


"""


class MCCmodel:
    def __init__(self, loadMode='axial', mode='math', v_ini=0.348 + 1., p_ini=-1.e5,
                 R_over=2.0, friction=30*np.pi/180,
                 nonlinearHardening=True, verboseFlag=True):
        # ---------------------------------------------------
        # network initialization if needed
        self.mode = mode
        if 'net' in self.mode or 'semi' in self.mode:
            try:
                self.vonMisesNet = Restore(
                    savedPath=os.path.join('misesModel', 'mdmddmd_Residual_Fourier_data_minmax'),
                )
                self.vonMisesNet.model.minmaxFlag = True
                self.hardeningNet = Restore(
                    savedPath=os.path.join('HardeningModel', 'mmmmd_Residual_Fourier_data_normalized'),
                )
            except:
                self.vonMisesNet = Restore(
                    savedPath=os.path.join('./FEMxEPxML/misesModel', 'mdmddmd_Residual_Fourier_data_minmax'),
                )
                self.vonMisesNet.model.minmaxFlag = True
                self.hardeningNet = Restore(
                    savedPath=os.path.join('./FEMxEPxML/HardeningModel', 'mmmmd_Residual_Fourier_data_normalized'),
                )
        # ---------------------------------------------------
        # material parameters
        self.nonlinearHardening = nonlinearHardening
        self.poisson = 0.3
        self.lambdaa = 0.1297
        self.kappa = 0.0322
        self.fric = friction
        # self.M_compression, self.M_extension = 6.*np.sin(self.fric)/(3.-np.sin(self.fric)), \
        #                                        6. * np.sin(self.fric) / (3. + np.sin(self.fric))
        self.M_compression, self.M_extension = 1.344, 1.344
        self.p0 = -1e5
        self.pc0 = R_over*self.p0  # largest pc in the history
        self.v_ini = v_ini         # the void ratio under pressure of p0 in natural state
        self.p_ini = p_ini         # the pressure at the start
        # void ratio and pressure at the beginning of the loading
        self.v = self.v_ini-self.lambdaa*np.log(self.pc0/self.p0)+self.kappa*np.log(self.pc0/self.p0)
        self.p = p_ini
        # bulk and shear modulus at the beginning of the loading
        self.K, self.G, self.lam = self.getKandGandLam(self.v, self.p)
        # material matrix at the beginning of the loading
        self.De = self.tangentAssemble(
            lam=self.lam,
            G=self.G)
        self.De_inv = np.linalg.inv(self.De)
        self.Dep = np.zeros(shape=[6, 6])
        self.M = np.array([[1., 0., 0., 0., 0., 0.],
                           [0., 1., 0., 0., 0., 0.],
                           [0., 0., 1., 0., 0., 0.],
                           [0., 0., 0., 0.5, 0., 0.],
                           [0., 0., 0., 0., 0.5, 0.],
                           [0., 0., 0., 0., 0., 0.5]
                           ])

        # ---------------------------------------------------
        # state variables [stress and strain vector in Voigt notion]
        self.sig = np.array([self.p, self.p, self.p, 0., 0., 0.])
        self.eps = np.zeros(6)
        self.eps_plastic_p = 0.
        self.eps_plastic_q = 0.
        self.sigTrial = np.zeros(6)
        self.lastYield = -1
        self.yieldValue = self.yieldFunction(p=self.p, q=0., pc0=self.pc0)
        self.loadHistoryList = [np.array(list(self.sig) + list(self.eps) +
                                         [self.eps_plastic_p, self.eps_plastic_q, self.pc0] + [self.yieldValue, 0])]

        # ---------------------------------------------------
        # load configuration
        self.verboseFlag = verboseFlag
        self.loadMode = loadMode  # 'axial' or 'random'
        if self.loadMode == 'random':
            self.epsAxialObject = 0.004  # random loading
        else:
            self.epsAxialObject = 0.01
        self.iterationNum = int(1e2)
        self.depsAxial = self.epsAxialObject / self.iterationNum

        # ---------------------------------------------------
        # Tolerance
        self.yieldToleranceNegtive = -100000.
        self.yieldTolerancePositive = 100000.
        if 'math' in self.mode:
            self.yieldToleranceNegtive = -self.K*0.01
            self.yieldTolerancePositive = self.K*0.01

    def getKandGandLam(self, v, p):
        K = v*np.abs(p) / self.kappa
        G = 1.5 * K * (1. - 2. * self.poisson) / (1 + self.poisson)
        lam = K - 2. / 3. * G
        return K, G, lam

    def forward(self, st=None, path=None, sampleIndex=None):
        mmax = np.max(np.abs(path))
        if mmax > 0.5:
            path = path / mmax * 0.5
        if self.loadMode == 'random':
            self.iterationNum = len(path)
        for i in range(self.iterationNum):
            if i == 42:
                print()
            print()
            print('Load step: %d' % i)
            if self.loadMode == 'random':  # load under the gaussian random loading path
                if i == 0:
                    deps = path[0]
                else:
                    # deps = 10*(path[i] - path[i - 1]) * self.epsAxialObject / np.max(np.abs(path))
                    deps = path[i] - path[i - 1]
                    deps = -deps
            elif st is not None:
                deps = st
            else:  # load under the conventional triaxial compression test
                deps = self.getAxialDeps()
                if 0.3 * self.iterationNum < i < 0.65 * self.iterationNum:
                    deps = - deps
            deps = np.array([deps[0], deps[1], 0., deps[2], 0., 0.])
            # if np.sum((self.eps+deps)[:2]) > 0.:
            #     deps = -deps
            iteration, sigTrial, materialMatrix, dEps_plastic_p, dEps_plastic_q, yieldValue, pc0 = self.solver(deps)
            # yieldValue = self.yieldFunction(self.sig)
            self.updateState(
                sig=sigTrial, deps=deps, yieldValue=yieldValue,
                dEps_plastic_p=dEps_plastic_p, dEps_plastic_q=dEps_plastic_q, pc0=pc0)

            self.loadHistoryList.append(np.array(list(self.sig) + list(self.eps) +
                                             [self.eps_plastic_p, self.eps_plastic_q, self.pc0] +
                                             [self.yieldValue, iteration]))
        self.plotMask(sampleIndex)

    def solver(self, deps):
        # update the material matrix of D
        sig = self.sig + self.De @ deps.T

        # used for debug ckeck  check the equation for p and q
        dsig = self.De @ deps.T
        dp, dq = getP(dsig), getQ(dsig)
        deps_p, deps_q = getVolStrain(deps), getQEps(deps)
        dpp, dqq = self.K*deps_p, 3.0*self.G*deps_q
        residual_check = dpp-dp, dqq-dq

        p, q = getP(sig), getQ(sig)
        # sig_principle = getPrinciple(sigTrial)
        # compressionFlag = True if all(sig_principle < 0) else False
        compressionFlag = True
        pc0 = self.pc0
        yieldValue = self.yieldFunction(p=p, q=q, pc0=pc0, compressionFlag=compressionFlag)
        iteration = 0
        elasticFlag = True
        r_mid = 1.
        dEps_plastic_p, dEps_plastic_q = 0., 0.

        # -----------------------------------
        # elastic
        if yieldValue <= 0:
            # self.v = self.v * (1 + getVolStrain(eps=deps * r_mid))
            # self.p = getP(self.sig)
            # self.K, self.G, self.lam = self.getKandGandLam(v=self.v, p=self.p)
            # self.De = self.tangentAssemble(lam=self.lam, G=self.G)
            materialMatrix = self.De

        # -----------------------------------
        # plastic and last step is elastic
        elif self.yieldValue < self.yieldToleranceNegtive:
            if self.verboseFlag:
                print()
                print('Bisection!!  yieldValue %.3f LastYieldValue %.3f' % (yieldValue, self.yieldValue))
                print()
            r_mid, yield_mid = self.transiformationSplit(deps, compressionFlag=compressionFlag)  # searching for the transit point
            sig = self.sig + np.dot(self.De, r_mid * deps)
            v = self.v*(1+getVolStrain(eps=deps*r_mid))
            depsLeft = (1 - r_mid) * deps
            iteration, sig, materialMatrix, dEps_plastic_p, dEps_plastic_q, yieldValue, pc0 = \
                self.plasticReturnMapping(deps=depsLeft, sigAfterBisection=sig, vLast=v)

        # -----------------------------------
        # last step is plastic
        else:
            iteration, sig, materialMatrix, dEps_plastic_p, dEps_plastic_q, yieldValue, pc0 = \
                self.plasticReturnMapping(deps=deps)

        return iteration, sig, materialMatrix, dEps_plastic_p, dEps_plastic_q, yieldValue, pc0

    # def getAveragedMatrix(self, materialMatrix, r_mid):
    #     temp = np.zeros(shape=(3, 3))
    #     for ix, iy in [[0, 0], [0, 1], [1, 0], [1, 1], [2, 2]]:
    #         temp[ix, iy] = 1. / (r_mid / self.D[ix, iy] + (1.0 - r_mid) / materialMatrix[ix, iy])
    #     return temp

    def updateState(self, sig, deps, yieldValue, dEps_plastic_p, dEps_plastic_q, pc0):
        self.sig = sig  # in Voigt notion (6)
        self.p = getP(sig)
        self.eps_plastic_p += dEps_plastic_p
        self.eps_plastic_q += dEps_plastic_q
        self.eps += deps
        self.yieldValue = yieldValue
        self.pc0 = pc0
        self.v = self.v*(1. + getVolStrain(deps))
        self.K, self.G, self.lam = self.getKandGandLam(self.v, self.p)
        self.De = self.tangentAssemble(lam=self.lam, G=self.G)

    def plotMask(self, sampleIndex):
        figTitle = 'Mises_%d_%s' % (
            self.iterationNum, self.loadMode + str(sampleIndex) if 'random' in self.loadMode else self.loadMode)
        if 'net' in self.mode or 'semi' in self.mode:
            figTitle = '%s_Mises_%d_%s' % (self.mode,
                                           self.iterationNum,
                                           self.loadMode + str(sampleIndex) if 'random' in self.loadMode else self.loadMode)
        if 'random' in self.loadMode:
            savePath = 'MCCData'
            figTitle = os.path.join('MCCresults', figTitle)
            writeDownPaths(
                path='./MCCData/MCCresults',
                data=np.array(self.loadHistoryList),
                sampleIndex=sampleIndex,
                mode=self.mode)
        else:
            savePath = 'figSav'
            figTitle = os.path.join('MisesBaseline', figTitle)
        plotHistory(loadHistory=self.loadHistoryList,
            figTitle=figTitle, savePath=savePath)

    def yieldFunction(self, q, p, pc0=None, compressionFlag=True):
        if compressionFlag:
            M = self.M_compression
        else:
            M = self.M_extension
        if pc0:
            yieldValue = q**2.+M**2.*p*(p-pc0)
        else:
            yieldValue = q**2.+M**2.*p*(p-self.pc0)
        return yieldValue

    def getHardening(self, epsPlastic):
        if 'net' in self.mode:
            hardingValue = self.hardeningNet.prediction(np.array([[epsPlastic]]))[0, 0]
        else:
            ' 500e6*(0.05+eplastic)**0.2'
            if self.nonlinearHardening:
                hardingValue = self.A * (self.epsilon0 + epsPlastic) ** self.n
            else:
                hardingValue = self.A * epsPlastic
        return hardingValue

    def getVonMises(self, sig):
        if 'net' in self.mode:
            vonMises = self.vonMisesNet.prediction(sig.reshape(1, 3))[0, 0]
        elif 'semi' in self.mode:
            vonMises = np.sqrt(sig[0] ** 2 - sig[0] * sig[1] + sig[1] ** 2 + 3. * sig[2] ** 2)
        else:  # 'math' in self.mode
            vonMises = np.sqrt(sig[0] ** 2 - sig[0] * sig[1] + sig[1] ** 2 + 3. * sig[2] ** 2)
        return vonMises

    def tangentAssemble(self, lam, G):
        D = np.zeros([6, 6])
        for i in range(3):
            for j in range(3):
                D[i, j] += lam
        D[0, 0] += 2. * G
        D[1, 1] += 2. * G
        D[2, 2] += 2. * G
        for i in range(3, 6):
            D[i, i] += G
        return D

    def getAxialDeps(self):
        dEps = np.array(
            [self.depsAxial, -self.De[1, 0] / self.De[1, 1] * self.depsAxial, 0.])
        return dEps

    def getDiffVectorOfYieldFunction(self, sig, pc0=None, compressionFlag=True):
        if 'net' in self.mode:
            mises, dmises = self.vonMisesNet.prediction2(sig.reshape(1, 3))
            dfdp, dfdq = dmises[0]
            dfdpc0 = -pc0
        else:
            if pc0 != None:
                pc00 = pc0
            else:
                pc00 = self.pc0
            p, q = getP(sig), getQ(sig)
            # decide whether in compression or extension state
            sig_principle = getPrinciple(sig)
            M = self.M_compression if all(sig_principle < 0.) else self.M_extension
            dfdp = M**2.*(2.*p-pc00)
            dfdq = 2*q
            dfdpc0 = -M**2.*p
        return dfdp, dfdq, dfdpc0

    def transiformationSplit(self, deps, compressionFlag):
        """
                Used to search the point where the loading
                transform into the plasticity from the ela-
                sticity.

        :return:
        """
        r_min, r_max = 0., 1.0
        r_mid = 0.5 * (r_min + r_max)
        # v = self.v*(1.+getVolStrain(eps=deps*r_mid))
        # self.K, self.G, self.lam = self.getKandGandLam(v, self.p)
        # self.D = self.tangentAssemble(self.lam, self.G)
        p_old, q_old = getP(self.sig), getQ(self.sig)
        yieldValue_old = self.yieldFunction(q=q_old, p=p_old, pc0=self.pc0)
        if yieldValue_old > self.yieldTolerancePositive:
            raise ValueError('The yield Value of the last elastic state is larger than 0.')
        sig = self.sig + np.dot(self.De, r_mid * deps)
        p, q = getP(sig), getQ(sig)
        yield_mid = self.yieldFunction(
            p=p, q=q, compressionFlag=compressionFlag)
        i = 0
        # while yield_mid < -self.yieldTolerance/10. or yield_mid > 0.:
        while yield_mid < self.yieldToleranceNegtive or yield_mid > self.yieldTolerancePositive:
            if yield_mid < self.yieldToleranceNegtive:
                r_min = r_mid
            else:
                r_max = r_mid
            r_mid = 0.5 * (r_min + r_max)
            sig = self.sig + np.dot(self.De, r_mid * deps)
            p, q = getP(sig), getQ(sig)
            yield_mid = self.yieldFunction(
                p=p, q=q, compressionFlag=compressionFlag)
            if i > 100 and i % 10 == 0:
                print('\titeration: %i yieldValue: %.3f rmid: %.3f' % (i, yield_mid, r_mid))
            i += 1
        return r_mid, yield_mid

    ''' 
    1. The extra components in x direction is sensible or not (this is right)
    2. Check the materialMatrix !!!
    '''

    def plasticReturnMapping(self, deps, sigAfterBisection=None, vLast=None):
        iteration = 0
        if sigAfterBisection is None:
            v0 = self.v
            p_old, q_old = getP(self.sig), getQ(self.sig)
            K0, G0, lam0, De0 = self.K, self.G, self.lam, self.De
            sig0 = self.sig
        else:
            v0 = vLast
            sig0 = sigAfterBisection
            p_old, q_old = getP(sig0), getQ(sig0)
            K0, G0, lam0, = self.getKandGandLam(v=v0, p=p_old)
            De0 = self.tangentAssemble(lam=lam0, G=G0)
        sigTrial = sig0+De0@deps
        pc0, pc0_safe = copy.deepcopy(self.pc0), copy.deepcopy(self.pc0)
        # sig_principle = getPrinciple(self.sig)
        # compressionFlag = True if all(sig_principle < 0.) else False
        compressionFlag = True
        dFdp_0, dFdq_0, dfdpc0_0 = self.getDiffVectorOfYieldFunction(
            sig=sig0, pc0=pc0,
            compressionFlag=compressionFlag)
        a, b = dFdp_0, dFdq_0
        v = v0 * (1 + getVolStrain(deps))

        p_trial, q_trial = getP(sigTrial), getQ(sigTrial)
        yieldValue = self.yieldFunction(p=p_trial, q=q_trial, compressionFlag=compressionFlag, pc0=pc0)
        # dfds_mat = np.zeros([3, 1])
        # H = 0.
        returnFlag = False
        dEps_plastic_p, dEps_plastic_q= 0., 0.
        """
                Yield surface correction scheme for general elastoplastic models:

            Reference: 
                [FLAC3D documents]
            1. http://docs.itascacg.com/3dec700/common/models/camclay/doc/modelcamclay.html#modelcamclay-3
        """
        while yieldValue > self.yieldTolerancePositive or yieldValue < self.yieldToleranceNegtive:
            returnFlag = True
            array_operator = np.array([1., 1., 1., 0., 0., 0.])
            # sigMidTrial = .5*(self.sig+sigTrial)
            p_trial, q_trial = getP(sigTrial), getQ(sigTrial)
            # sig_principle = getPrinciple(sigTrial)
            # compressionFlag = True if all(sig_principle < 0.) else False
            compressionFlag = True
            '''
                NOTE: Problems for error continuing increasing may arise here, since we use the pc0 at last state to 
                    calculate the differentiation.
            '''
            # dFdp, dFdq, dfdpc0 = self.getDiffVectorOfYieldFunction(
            #     sig=sigTrial, pc0=pc0, compressionFlag=compressionFlag)

            # solve for the dLambda
            # decide whether in compression or extension state
            M = self.M_compression if compressionFlag else self.M_extension
            '''
            a = (M*K0)**2.*dFdp_0**2.+(3.*G0)**2.*dFdq_0**2.
            b = -(K0*dFdp*dFdp_0+3.*G0*dFdq*dFdq_0)
            c = yieldValue
            '''
            temp = v0/(self.lambdaa-self.kappa)
            a_ = (3.*G0*b)**2.+(M*K0*a)**2.-M**2.*K0*a**2.*pc0*temp
            # b_ = -2.*q_trial*b*3.*G0-M**2.*2.*p_trial*K0*a+M**2.*p_trial*pc0*a*temp+M**2.*K0*a*pc0
            b_ = -2.*q_trial*b*3.*G0+M**2.*(-2.*p_trial*K0*a+p_trial*pc0*a*temp+K0*a*pc0)
            c_ = q_trial**2.+M**2.*p_trial*(p_trial-pc0)

            dLambda = np.min(np.poly1d([a_, b_, c_]).r)
            if isinstance(dLambda, complex):
                raise ValueError('No real solution of the quadratic equation!! Please ckeck!!')
            dEps_plastic_p = dLambda*dFdp_0
            dEps_plastic_q = dLambda*dFdq_0

            dpeps = getVolStrain(deps)
            dqeps = getQEps(deps)

            p_new = p_trial - K0*dEps_plastic_p
            q_new = q_trial - 3.*G0*dEps_plastic_q
            s_trial = sigTrial - p_trial * array_operator
            s_new = s_trial * q_new/q_trial
            sigTrial_new = s_new + p_new*array_operator
            sigTrial = sigTrial_new

            '''
                NOTE: Problems may come from here, since only $x$ is small can $x$ be used to approximate $\ln(1+x)$
            '''

            # pc0 = pc0 * (1. - dEps_plastic_p * v0 / (self.lambdaa - self.kappa))
            pc0 = q_new**2/M**2/p_new+p_new

            # ckeck = q_new ** 2 + M**2*p_new*(p_new-(q_new**2/M**2/p_new+p_new))

            # hardening = self.getHardening(epsPlastic=epsPlastic)
            yieldValue = self.yieldFunction(p=p_new, q=q_new, pc0=pc0, compressionFlag=compressionFlag)

            iteration += 1
            if self.verboseFlag:
                print('iteration: %d yieldValue: %.8f pc0: %.5e eps: %.3e %.3e %.3e dep: %.3e p: %.3e q: %.3e K: %.3e' %
                      (iteration, yieldValue, pc0, self.eps[0], self.eps[1], self.eps[3], dEps_plastic_p, p_new, q_new, self.K))
            if iteration >= 20:
                # if yieldValue < 0:
                #     break
                self.plotMask(sampleIndex=-1)
                raise ValueError('Iteration number exceeds!!!')
            break  # according to the function 27, only 1 iteration is needed to get the proper dlambda
        if np.abs(yieldValue) > self.yieldTolerancePositive:
            raise ValueError('The yieldValue (%.5e) is more than the yield tolerance (%.5e).' %
                             (yieldValue, self.yieldTolerancePositive))
        '''
            Mark: this is where the bug comes from
                
                `materialMatrix = A/H*self.D`
        '''

        '''
            TODO add the MATERIAL MATRIX calculation to implement 
                the Newtown-Raphson non-linear iteration in FEM
        '''

        #
        dpc0 = - dEps_plastic_p * v0 / (self.lambdaa - self.kappa)*pc0
        dFdp, dFdq, dFdpc0 = self.getDiffVectorOfYieldFunction(sig=sig0, pc0=pc0_safe, compressionFlag=True)
        if returnFlag:
            '''
                Reference: 
                [1] Cheng Mingxiang, Elastoplastic mechanics, Page 264. 
            '''
            h = -dFdpc0*dpc0/dLambda
            dpdsigma, dqdsigma = get_dpdsig(sigTrial)
            dfdsigma = dFdp*dpdsigma+dFdq*dqdsigma
            K, G, lam = self.getKandGandLam(v0, p_old)
            D = self.tangentAssemble(lam, G)
            H = h+dfdsigma.reshape([1, -1])@D@dfdsigma.reshape([-1, 1])
            materialMatrix = D - D@dfdsigma.reshape([-1, 1])@dfdsigma.reshape([1, -1])@D/H[0, 0]
        else:
            materialMatrix = self.De
        return iteration, sigTrial, materialMatrix, dEps_plastic_p, dEps_plastic_q, yieldValue, pc0


def getP(sigma):
    if len(sigma.shape) == 1:
        return np.average(sigma[:3])
    else:
        return np.average(sigma[:, :3], axis=1).reshape(-1, 1)


def getS(sigma):
    return sigma-np.array([1., 1., 1., 0., 0., 0.])*getP(sigma)


def getJ2(sigma):
    s = getS(sigma)
    if len(sigma.shape) == 1:
        return 0.5 * np.sum(s**2*np.array([1., 1., 1., 2., 2., 2.]))
    elif len(sigma.shape) == 2:
        return 0.5 * np.sum(s ** 2 * np.array([1., 1., 1., 2., 2., 2.]), axis=1)


def getQ(sigma):
    J2 = getJ2(sigma)
    return np.sqrt(3. * J2)


def getVolStrain(eps):
    if len(eps.shape) == 1:
        return np.sum(eps[..., :3])
    elif len(eps.shape) == 2:
        return np.sum(eps[..., :3], axis=1).reshape(-1, 1)
    else:
        raise ValueError('Shape of the eps is (%d, %d)' % eps.shape)


def getEpsDevitoric(eps):
    return eps - np.array([1., 1., 1., 0., 0., 0.])*getVolStrain(eps)/3.


def getJ2Eps(eps):
    e = getEpsDevitoric(eps)
    if len(eps.shape) == 1:
        return 0.5 * np.sum(e**2*np.array([1., 1., 1., 0.5, 0.5, 0.5]))
    elif len(eps.shape) == 2:
        return 0.5 * np.sum(e ** 2 * np.array([1., 1., 1., 0.5, 0.5, 0.5]), axis=1)
    else:
        raise ValueError('Shape of the eps is (%d, %d)' % eps.shape)


def getQEps(eps):
    J2eps = getJ2Eps(eps)
    return 2. / 3. * np.sqrt(3. * J2eps)


def getPrinciple(sigma):
    # if stress in voigt notion
    sigma_matrix = np.array([[sigma[0], sigma[3], 0.],
                             [sigma[3], sigma[1], 0.],
                             [0., 0., sigma[2]]])
    # U1, sig_principle, U2 = np.linalg.svd(sigma_matrix)
    # sig_diag = np.diag(sig_principle)
    # sig1 = U1 @ sig_diag @ U2
    return np.linalg.eigvals(sigma_matrix)[:2]


def get_dpdsig(sigma):
    s = getS(sigma)
    p, q = getP(sigma), getQ(sigma)
    temp = 1./3.
    dpdsig = np.array([temp, temp, temp, 0., 0., 0.])
    dqdJ2 = 1.5/q if q != 0. else 1.
    dJ2ds = s*np.array([1., 1., 1., 2., 2., 2.])
    dsdsigma = np.array([1., 1., 1., 1., 1., 1.])
    dqdsigma = dqdJ2*dJ2ds*dsdsigma
    return dpdsig, dqdsigma


def plotHistory(loadHistory, dim=3, vectorLen=6, figTitle=None, savePath='./figSav'):
    '''
    np.array(list(self.sig) + list(self.eps) +
    [self.eps_plastic_p, self.eps_plastic_q, self.pc0] + [self.yieldValue, 0])
    '''
    load_history = np.array(loadHistory)
    sig = load_history[..., :vectorLen]
    eps = load_history[..., vectorLen:vectorLen * 2]
    epsPlasticVector = load_history[..., (vectorLen * 2):(vectorLen * 2 + 2)]
    pc0 = load_history[..., vectorLen * 2+2]
    yieldVlue = load_history[..., vectorLen * 2+3]
    iteration = load_history[..., vectorLen * 2+4]

    plt.figure(figsize=(16, 7))
    # strain
    ax = plt.subplot(231)
    epsLabel = ['$\epsilon_{xx}$', '$\epsilon_{yy}$', '$\epsilon_{xy}$'] if dim == 2 else \
        ['$\epsilon_{xx}$', '$\epsilon_{yy}$', '$\epsilon_{zz}$', '$\epsilon_{xy}$', '$\epsilon_{yz}$',
         '$\epsilon_{xz}$']
    plotSubFigures(ax, x=[range(len(eps)) for _ in range(len(eps[0]))], y=eps.T,
        label=epsLabel,
        xlabel='Load step', ylabel='$\epsilon$', num=vectorLen)

    # yield Value
    ax = plt.subplot(232)
    yieldVlue = yieldVlue.reshape(-1)
    plotSubFigures(ax=ax, x=range(len(sig)), y=yieldVlue, label='yieldValue', xlabel='Load step', ylabel='yieldValue')
    # plt.yscale('log')
    # plt.ylim([np.min(yieldVlue), np.max(yieldVlue)])
    ax2 = ax.twinx()
    ax2.plot(range(len(sig)), iteration, label='iterationNum', color='r', marker='o', lw=3)
    plt.ylabel('iterationNum', fontsize=12)
    # plt.ylim([-0.5, 8.0])
    plt.legend(fontsize=15)
    plt.yticks(fontsize=12)

    # stress
    ax = plt.subplot(233)
    sigLabel = ['$\sigma_{xx}$', '$\sigma_{yy}$', '$\sigma_{xy}$'] if dim == 2 else \
        ['$\sigma_{xx}$', '$\sigma_{yy}$', '$\sigma_{zz}$', '$\sigma_{xy}$', '$\sigma_{yz}$',
         '$\sigma_{xz}$']
    plotSubFigures(ax, x=[range(len(sig)) for _ in range(len(sig[0]))], y=sig.T,
        label=sigLabel,
        xlabel='Load step', ylabel='$Pa$', num=vectorLen)

    # plastic strain
    ax = plt.subplot(234)
    epsVol = getVolStrain(eps).reshape(-1, 1)
    e = getQEps(eps).reshape(-1, 1)
    epsPlasticVector = np.concatenate((epsPlasticVector, epsVol, e), axis=1)
    epsLabelPlastic = ['$\epsilon_{p}^p$', '$\epsilon_{q}^p$', '$\epsilon_{p}^e$', '$\epsilon_{q}^e$']
    plotSubFigures(ax, x=[range(len(epsPlasticVector)) for _ in range(len(epsPlasticVector[0]))], y=epsPlasticVector.T,
        label=epsLabelPlastic,
        xlabel='Load step', ylabel='$\epsilon$', num=4)

    # plot the q-p plane
    ax = plt.subplot(235)
    p = getP(sig).reshape(-1)/1e6
    q = getQ(sig).reshape(-1)/1e6
    q_crit = np.abs(p)*0.6556237707286147
    epsLabelPlastic = ['$q$', '$q_{crit}$']

    plotSubFigures(ax, x=[p, p], y=[q, q_crit],
                   label=epsLabelPlastic,
                   xlabel='p (MPa)', ylabel='q (MPa)', num=2)

    plt.tight_layout()
    fname = './%s/%s.png' % (savePath, figTitle if figTitle else 'Mises')
    plt.savefig(fname, dpi=200)
    plt.close()
    print('Figrue save as %s\n\n' % fname)
    return


def writeDownPaths(path, sampleIndex, data, mode):
    """
    np.array(list(self.sig) + list(self.eps) +
                                                 [self.vonMises, self.epsPlastic, self.hardening] +
                                                 list(self.epsPlasticVector) + [self.yieldValue, iteration])
    :param path:
    :param sampleIndex:
    :param data:
    :return:
    """
    if 'net' in mode:
        name = 'Net_random_%d.dat' % sampleIndex
    elif 'semi' in mode:
        name = 'Semi_random_%d.dat' % sampleIndex
    else:
        name = 'random_%d.dat' % sampleIndex
    filePath = os.path.join(path, name)
    np.savetxt(fname=filePath, X=data, fmt='%10.5f', delimiter=',',
        header='sigma_xx, sigma_yy, sigma_xy, epsilon_xx, epsilon_yy, epsilon_xy, ' +
               'vonMises, epsPlastic, hardening, ' +
               'epsilonP__xx, epsilonP__yy, epsilonP__xy, yieldValue, iteration')


# --------------------------------------------
# main
# load path reader
if __name__ == '__main__':
    baselineFlag = False
    mode = 'math'  # math net semi
    if not baselineFlag:
        # ----------------------------------------
        # training data generation
        loadPathList = loadingPathReader()[1:2]
        print()
        print('=' * 80)
        print('\t Path loading ...')
        for i in range(len(loadPathList)):
            print('\t\tPath %d' % i)
            mises = MCCmodel(loadMode='random', mode=mode, nonlinearHardening=True)
            mises.forward(path=loadPathList[i], sampleIndex=i)
    else:
        # ----------------------------------------
        # training data generation  (in conventional triaxial loading mode)
        mises = MCCmodel(loadMode='axial', nonlinearHardening=True)
        mises.forward()


