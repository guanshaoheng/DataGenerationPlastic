import os
import sys
import matplotlib.pyplot as plt
from misesTraining import Net, modelTrainning, Restore, findDevice, pickle_dump, pickle_load, saveScalar, dataReader
import numpy as np
import torch
import time


if __name__ == "__main__":
    if not os.path.exists('HardeningModel'):
        os.mkdir('HardeningModel')
    layerList = 'dmmdmd'
    savedPath = os.path.join('HardeningModel', layerList)
    if not os.path.exists(savedPath):
        os.mkdir(savedPath)

    device = findDevice()

    epsPlas, H, dHdEps = dataReader(root_path=savedPath, readContent='epsPlastic')


    sampleNum = 1000
    index_random = np.random.permutation(range(len(epsPlas)))
    epsPlas, H, dHdEps = epsPlas[index_random[:sampleNum]], \
                             H[index_random[:sampleNum]], dHdEps[index_random[:sampleNum]]

    # ------------------------------------------------
    # model training
    misesNet = Net(inputNum=1, outputNum=1, layerList=layerList).to(device=device)
    mTrain = modelTrainning(misesNet, savedPath=savedPath, device=device, normalizationFlag=True)
    mTrain.trainModel(x=epsPlas, y=H, dy=dHdEps)

    # restore
    model_evaluation = Restore(savedPath=savedPath, device=device, normalizationFlag=True)
    model_evaluation.evaluation(x=epsPlas, y=H, dy=dHdEps)
    print()

