import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class GuanssianRandomPath:
    """
        Used for random loading path generation via Gaussian Process method
    """
    def __init__(self, curlDegree, amplitudeValue, showFlag=False, maxEpsilonLimitation=0.8,
                 numberPerSamples=3, numberOfPoints=1000, meanValue=-1e5):
        self.seed = 10001
        np.random.seed(self.seed)

        self.showFlag = showFlag
        self.maxEpsilonLimitation = maxEpsilonLimitation
        self.numberOfPoints = numberOfPoints
        self.numberOfFuncutions = numberPerSamples

        self.curlDegree = curlDegree  # 1~5
        self.amplitudeValue = amplitudeValue  # generally 0.25
        self.amplitude = np.linspace(0, self.amplitudeValue, int(self.numberOfPoints))
        self.meanValue = -meanValue
        if meanValue!=0:
            self.x = np.abs(self.meanValue)*self.curlDegree*np.linspace(0, 1., self.numberOfPoints)[:, np.newaxis]
        else:
            self.x = self.curlDegree*np.linspace(0, 1., self.numberOfPoints)[:, np.newaxis]
        self.cov = self.CovarienceMatrix(self.x, self.x)*self.amplitude

        self.y = np.random.multivariate_normal(mean=np.ones(self.numberOfPoints)*self.meanValue,
                                               cov=self.cov,
                                               size=self.numberOfFuncutions)

        self.confiningPreussure = []
        if self.showFlag:
            self.plotPaths()
            self.plotCovarianceMatrix(kernel=self.cov, curl=self.curlDegree)

        # if self.generatingNum > 0:
        #     self.generation()

    def generation(self, generatingNum, path):
        print()
        print('='*80)
        print('\t Loading path generation ...')
        i = 0
        numSample = 0
        while numSample < generatingNum:
            print('\t\tPath random % d seed %d' % (numSample, i))
            self.seed = i
            np.random.seed(self.seed)
            self.y = np.random.multivariate_normal(
                mean=np.ones(self.numberOfPoints)*self.meanValue,
                cov=self.cov,
                size=self.numberOfFuncutions)
            maxEpsilon = np.max(np.abs(self.y))
            if maxEpsilon > self.maxEpsilonLimitation and self.numberOfFuncutions == 3:
                i += 1
                continue
            else:
                if self.numberOfFuncutions == 3:
                    self.plotPaths(path=path)
                    self.writeDownPaths(numSample, self.curlDegree)
                else:
                    self.plotPaths(path=path)
                    self.confiningPreussure.append(self.y.T)
                i += 1
                numSample += 1
        if self.numberOfFuncutions == 1:
            self.writeDownPaths(numSample, curlDegree=self.curlDegree)

    def CovarienceMatrix(self, x, y):
        """
            Use the kernel fucntion: $\kappa(x_i, x_j)=\mathrm{exp}(-\sum_{k=1}^{m}\theta_k(x_i^k-x_j^k)^2))$
                where the dimensional number is 1 in this project.

            Reference:
                [1] https://blog.dominodatalab.com/fitting-gaussian-process-models-python
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
        # interval = np.linspace(0.001, 1., 1000)
        # x_tick = interval
        # y_tick = interval
        # data = kernel
        # pd_data = pd.DataFrame(data, index=y_tick, columns=x_tick)
        # ax = sns.heatmap(pd_data, xticklabels=50, cmap="YlGnBu")
        ax = sns.heatmap(kernel, xticklabels=[], yticklabels=[], cmap="YlGnBu")
        plt.title('Degree of curl = %.2f' % curl)

        plt.tight_layout()
        if self.numberOfFuncutions == 3:
            plt.savefig('./figSav/CovariabceHeatMap_curl%d_new.png' % curl, dpi=200)
        else:
            plt.savefig('./ConfiningPressure/CovariabceHeatMap_curl%d_new.png' % curl, dpi=200)
        plt.close()

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

    def writeDownPaths(self, numSample, curlDegree):
        if self.numberOfFuncutions == 3:
            filePath = './vonmisesPaths/path_curlDegree%d_%d.dat' % (numSample, curlDegree)
            np.savetxt(fname=filePath, X=self.y.T, fmt='%10.5f', delimiter=',', header='epsilon_xx, epsilon_yy, epsilon_xy')
        elif self.numberOfFuncutions == 1:
            filePath = './ConfiningPressure/ConfiningPressurePath_curlDegree%d.dat' % curlDegree
            np.savetxt(fname=filePath, X=self.y.T, fmt='%10.5f', delimiter=',', header=' '.join(['%d' % i for i in range(numSample)]))


if __name__ == "__main__":
    # confining pressure generation
    # gaussian = GuanssianRandomPath(curlDegree=2, amplitudeValue=0.15, showFlag=True, numberPerSamples=3, meanValue=-1e5,
    #                                numberOfPoints=100)  # generally 1~5, 0.25
    # gaussian.generation(generatingNum=10, path='ConfiningPressure')

    # loading path for von-mises model
    for curlDegree in range(1, 6):
        gaussian = GuanssianRandomPath(curlDegree=curlDegree, amplitudeValue=0.15, showFlag=True, numberPerSamples=3, meanValue=0.0,
                                       numberOfPoints=1000)  # generally 1~5, 0.25
        gaussian.generation(generatingNum=200, path='vonmisesPaths')


