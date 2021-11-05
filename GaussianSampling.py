import scipy
import scipy.spatial as T
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


# Define the exponentiated quadratic
def exponentiated_quadratic(xa, xb):
    """Exponentiated quadratic  with σ=1"""
    # L2 distance (Squared Euclidian)
    sq_norm = -0.5 * T.distance.cdist(xa, xb, 'sqeuclidean')
    return np.exp(sq_norm)


# set the random seed
seed = 10001
np.random.seed(seed)
# Sample from the Gaussian process distribution
nb_of_samples = 100  # Number of points in each function
number_of_functions = 5  # Number of functions to sample
# Independent variable samples
X = np.linspace(-4, 4, nb_of_samples).reshape(-1, 1)
cov = exponentiated_quadratic(X, X)  # Kernel of data points
meanValue = -1e5
# Draw samples from the prior at our data points.
# Assume a mean of 0 for simplicity
# ys = np.random.multivariate_normal(
#     mean=np.zeros(nb_of_samples), cov=cov,
#     size=number_of_functions)
ys = np.random.multivariate_normal(
    mean=np.ones(nb_of_samples) * meanValue, cov=cov*pow(meanValue/3, 2),
    size=number_of_functions)
np.savetxt('./ConfingPressureGP_%d.csv' % seed, ys.T, delimiter=' ')

# Plot the sampled functions
fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=100)
for i in range(number_of_functions):
    plt.plot(list(range(1, len(ys[0])+1))[::2], list(ys[i]/1e6)[::2], linestyle='-', marker='o', markersize=4)
plt.xlabel('Loading step', fontsize=15)
plt.ylabel('$\sigma_{11}$ (MPa)', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(ticks=[-0.16, -0.13, -0.10, -0.07, -0.04], fontsize=15)
# plt.locator_params(axis='y', nbins=5)
# Index locator
# ax.yaxis.set_major_locator(ticker.IndexLocator(base=0.1, offset=0.25))
# ax.yaxis.set_major_locator(ticker.AutoLocator())
# ax.yaxis.set_major_locator(ticker.AutoMinorLocator())
# plt.title((
#         '%d different function realizations at %d points sampled from\n' % (number_of_functions, nb_of_samples) +
#         ' a Gaussian process with exponentiated quadratic kernel'))
plt.xlim([0, len(ys[0])])
plt.tight_layout()
# plt.show()
plt.savefig('./ConfiningPressureGP_%d.svg' % seed)


def plotCovarianceMatrix():
    # Show covariance matrix example from exponentiated quadratic
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), dpi=100)
    xlim = (-3, 3)
    X = np.expand_dims(np.linspace(*xlim, 25), 1)
    covarianceMatrix = exponentiated_quadratic(X, X)
    # Plot covariance matrix
    im = ax1.imshow(covarianceMatrix)  # cmap=cm.YlGnBu)
    cbar = plt.colorbar(
        im, ax=ax1, fraction=0.045, pad=0.05)
    cbar.ax.set_ylabel('$k(x,x)$', fontsize=10)
    ax1.set_title((
        'Exponentiated quadratic \n'
        'example of covariance matrix'))
    ax1.set_xlabel('x', fontsize=13)
    ax1.set_ylabel('x', fontsize=13)
    ticks = list(range(xlim[0], xlim[1] + 1))
    ax1.set_xticks(np.linspace(0, len(X) - 1, len(ticks)))
    ax1.set_yticks(np.linspace(0, len(X) - 1, len(ticks)))
    ax1.set_xticklabels(ticks)
    ax1.set_yticklabels(ticks)
    ax1.grid(False)

    # Show covariance with X=0
    xlim = (-4, 4)
    X = np.expand_dims(np.linspace(*xlim, num=100), 1)
    zero = np.array([[0]])
    covarianceMatrix0 = exponentiated_quadratic(X, zero)
    # Make the plots
    ax2.plot(X[:, 0], covarianceMatrix0[:, 0], label='$k(x,0)$')
    ax2.set_xlabel('x', fontsize=13)
    ax2.set_ylabel('covariance', fontsize=13)
    ax2.set_title((
        'Exponentiated quadratic  covariance\n'
        'between $x$ and $0$'))
    # ax2.set_ylim([0, 1.1])
    ax2.set_xlim(*xlim)
    ax2.legend(loc=1)

    fig.tight_layout()
    plt.show()
