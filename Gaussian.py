import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class GuanssianRandomPath:
    def __init__(self, curlDegree, amplitudeValue, seed=10001, showFlag=False):
        self.seed = seed
        np.random.seed(self.seed)

        self.showFlag = showFlag
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
        self.writeDownPaths()

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

    def plotPaths(self):
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
        # plt.title((
        #         '%d different function realizations at %d points sampled from\n' % (number_of_functions, nb_of_samples) +
        #         ' a Gaussian process with exponentiated quadratic kernel'))
        plt.xlim([0, len(self.y[0])])
        plt.tight_layout()
        plt.legend()
        plt.savefig('./figSav/ConfiningPressureGP_curl%d.png' % self.curlDegree, dpi=200)

    def writeDownPaths(self):
        filePath = './MCCData/path_%d.dat' % self.seed
        np.savetxt(fname=filePath, X=self.y.T, fmt='%10.5f', delimiter=',', header='epsilon_xx, epsilon_yy, epsilon_xy')


# gaussian = GuanssianRandomPath(curlDegree=5, amplitudeValue=0.25)  # generally 1~5, 0.25

for i in range(100):
    gaussian = GuanssianRandomPath(curlDegree=2, amplitudeValue=0.25)


