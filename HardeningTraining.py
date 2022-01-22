import os
import sys
import matplotlib.pyplot as plt
from misesTraining import Net, modelTrainning, Restore, findDevice, pickle_dump, pickle_load, saveScalar, dataReader
import numpy as np
import torch
import time
from misesTraining import SSP
from torch import nn


if __name__ == "__main__":
    layerList = 'mmmmd'
    residualFlag = False
    savedPath = os.path.join('HardeningModel', 'ResidualNet' if residualFlag else layerList)
    functionGeneration = False
    normalizationFlag = False
    fourier_features = True
    savedPath += ('_Fourier' if fourier_features else '_noFourier')
    savedPath += ('_generation' if functionGeneration else '_data')
    savedPath += ('_scaled' if normalizationFlag else '_Noscaled')
    print()
    print('-' * 80)
    print(savedPath)
    if not os.path.exists(savedPath):
        os.mkdir(savedPath)
    if functionGeneration:
        epsPlas, H, dHdEps = dataReader(root_path=savedPath, readContent='epsPlastic')
    else:
        epsPlas, H, dHdEps = dataReader(root_path=savedPath, readContent='epsPlastic')

    sampleNum = 5000
    index_random = np.random.permutation(range(len(epsPlas)))
    x, y, dy = epsPlas[index_random[:sampleNum]], \
                             H[index_random[:sampleNum]], dHdEps[index_random[:sampleNum]]

    # ------------------------------------------------
    device = findDevice()
    # model training
    x_min, x_max, y_min, y_max, dy_min, dy_max, y_std, dy_std = \
        pickle_load('x_min', 'x_max', 'y_min', 'y_max', 'dy_min', 'dy_max', 'y_std', 'dy_std', root_path=savedPath)
    if residualFlag:
        # misesNet = DenseResNet(xmin=x_min, xmax=x_max, ymin=y_min, ymax=y_max, device=device,
        #                    inputNum=3, outputNum=1, fourier_features=fourier_features).to(device=device)
        misesNet = Net(xmin=x_min, xmax=x_max, ymin=y_min, ymax=y_max, device=device,
                       activation=nn.Sigmoid(), inputNum=1, outputNum=1, layerList=layerList, node=100,
                       fourier_features=fourier_features, m_freqs=50).to(device=device)
    else:
        misesNet = Net(xmin=x_min, xmax=x_max, ymin=y_min, ymax=y_max, device=device,
                       activation=nn.Sigmoid(), inputNum=1, outputNum=1, layerList=layerList, node=100,
                       fourier_features=fourier_features, m_freqs=50).to(device=device)
    mTrain = modelTrainning(model=misesNet, savedPath=savedPath,
                            device=device, optimizerSTR='adam',
                            normalizationFlag=normalizationFlag,
                            epochMax=int(1e6), batchSize=0, dyWeight=10. * (y_std[0] / dy_std[0]) ** 2, patienceNum=100)
    mTrain.trainModel(x=x, y=y, dy=dy)

    # restore
    # model_evaluation = Restore(savedPath=os.path.join(savedPath, 'epoch_3000'),
    #                            device=device, normalizationFlag=normalizationFlag)
    model_evaluation = Restore(savedPath=savedPath, device=device, normalizationFlag=normalizationFlag)
    model_evaluation.evaluation(x=x, y=y, dy=dy)
    print()
