import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class GuanssianRandomPath:
    """
        Used for random loading path generation via Gaussian Process method
    """
    def __init__(self, curlDegree, amplitudeValue, showFlag=False, generatingNum=0, maxEpsilonLimitation=0.8):
        self.seed = 10001
        np.random.seed(self.seed)

        self.showFlag = showFlag
        self.generatingNum = generatingNum
        self.maxEpsilonLimitation = maxEpsilonLimitation
        self.numberOfPoints = 1000
        self.numberOfFuncutions = 3

        self.curlDegree = curlDegree  # 1~5
        self.amplitudeValue = amplitudeValue  # generally 0.25
        self.amplitude = np.linspace(0, self.amplitudeValue, int(self.numberOfPoints))
        self.x = self.curlDegree*np.linspace(0, 1., self.numberOfPoints)[:, np.newaxis]
        self.meanValue = -0.
        self.cov = self.CovarienceMatrix(self.x, self.x)*self.amplitude

        self.y = np.random.multivariate_normal(mean=np.ones(self.numberOfPoints)*self.meanValue,
                                               cov=self.cov,
                                               size=self.numberOfFuncutions)
        if self.showFlag:
            self.plotPaths()
            self.plotCovarianceMatrix(kernel=self.cov, curl=self.curlDegree)

        if self.generatingNum > 0:
            self.generation()

    def generation(self):
        print()
        print('='*80)
        print('\t Loading path generation ...')
        i = 0
        numSample = 0
        while numSample < self.generatingNum:
            print('\t\tPath random % d seed %d' % (numSample, i))
            self.seed = i
            np.random.seed(self.seed)
            self.y = np.random.multivariate_normal(mean=np.ones(self.numberOfPoints)*self.meanValue,
                                               cov=self.cov,
                                               size=self.numberOfFuncutions)
            maxEpsilon = np.max(np.abs(self.y))
            if maxEpsilon > self.maxEpsilonLimitation:
                i += 1
                continue
            else:
                # self.plotPaths(path='MCCData')
                self.writeDownPaths(numSample)
                i += 1
                numSample += 1

    def CovarienceMatrix(self, x, y):
        """
            Use the kernel fucntion: $\kappa(x_i, x_j)=\mathrm{exp}(-\sum_{k=1}^{m}\theta_k(x_i^k-x_j^k)^2))$
                where the dimensional number is 1 in this project.
        :param x:
        :param y:
        :return:
        """
        mesh = np.meshgrid(x, y)
        kernel = np.exp(-(mesh[0]-mesh[1])**2)
        return kernel

    def plotCovarianceMatrix(self, kernel, curl):
        numberOfticksInFigure = 11
        interval = int(len(kernel)/numberOfticksInFigure)
        ax = sns.heatmap(kernel, xticklabels=interval, yticklabels=interval, cmap="YlGnBu")
        plt.title('Degree of curl = %.2f' % curl)
        plt.tight_layout()
        plt.savefig('./figSav/CovariabceHeatMap_curl%d.png' % curl, dpi=200)

    def plotPaths(self, path='figSav'):
        # Plot the sampled functions
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=100)
        totalPointsOnFigure = 50
        interval = int(len(self.y[0])/50)
        labelList = ['xx', 'yy', 'xy']
        for i in range(self.numberOfFuncutions):
            plt.plot(list(range(1, len(self.y[0]) + 1))[::interval], list(self.y[i])[::interval],
                     linestyle='-', marker='o', markersize=4, label='$\epsilon_{%s}$' % labelList[i])
        plt.xlabel('Loading step', fontsize=15)
        plt.ylabel('$\epsilon$', fontsize=15)
        plt.xticks(fontsize=15)
        plt.xlim([0, len(self.y[0])])
        plt.tight_layout()
        plt.legend()
        figName = 'ConfiningPressureGP_curl%d_seed%d.png' % (self.curlDegree, self.seed)
        plt.savefig(os.path.join(path, figName), dpi=200)
        plt.close()

    def writeDownPaths(self, numSample):
        filePath = './MCCData/path_%d.dat' % numSample
        np.savetxt(fname=filePath, X=self.y.T, fmt='%10.5f', delimiter=',', header='epsilon_xx, epsilon_yy, epsilon_xy')


gaussian = GuanssianRandomPath(curlDegree=2, amplitudeValue=0.15, generatingNum=50)  # generally 1~5, 0.25



