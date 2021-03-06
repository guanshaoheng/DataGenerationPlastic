import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
from torch import nn
import pickle
import tensorflow as tf
from networkTF import TensorFlowModel


def getH(epsPlastic):
    H = 500e6 * (0.05 + epsPlastic) ** 0.2
    return H


def get_dHdEps(epsPlastic):
    dHdEps = -500e6 * 0.2 * (0.05 + epsPlastic) ** (0.2 - 1.)
    return dHdEps


def Hgeneration(root_path):
    epsPlas = np.random.random(size=(1000, 1))
    H = getH(epsPlas).reshape(-1, 1)
    dHdEps = np.array([get_dHdEps(i) for i in epsPlas]).reshape(-1, 1)

    saveScalar(x=epsPlas, y=H, dy=dHdEps, root_path=root_path)
    return epsPlas, H, dHdEps


def getMeanStd(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    mmin = np.min(data, axis=0)
    mmax = np.max(data, axis=0)
    return mean, std, mmin, mmax


def saveScalar(x, y, dy, root_path):
    x_mean, x_std, x_min, x_max = getMeanStd(x)
    y_mean, y_std, y_min, y_max = getMeanStd(y)
    dy_mean, dy_std, dy_min, dy_max = getMeanStd(dy)
    pickle_dump(root_path=root_path,
                x_mean=x_mean, x_std=x_std, x_min=x_min, x_max=x_max,
                y_mean=y_mean, y_std=y_std, y_min=y_min, y_max=y_max,
                dy_mean=dy_mean, dy_std=dy_std, dy_min=dy_min, dy_max=dy_max)
    return


def minMaxLinear(x, xmin, xmax):
    normed = (x - xmin) / (xmax-xmin)
    return normed.astype(np.float32)


def minMaxLinearReversed(x, xmin, xmax):
    revesed = x * (xmax-xmin) + xmin
    return revesed.astype(np.float32)


def normalize(x, xmean, xstd):
    normed = (x -xmean) / xstd
    return normed.astype(np.float32)


def normalizeReverse(x, xmean, xstd):
    reveresed = x * xstd + xmean
    return reveresed


def pickle_dump(**kwargs):
    root_path = kwargs['root_path']
    savePath = os.path.join(root_path, 'scalar')
    try:
        os.mkdir(savePath)
    except:
        pass
    for k in kwargs:
        if k != 'root_path':
            f = open(os.path.join(savePath, '%s' % k), 'wb')
            pickle.dump(kwargs[k], f, 0)
            f.close()


def pickle_load(*args, root_path):
    cwd = os.getcwd()
    if 'sciptes4figures' in cwd:
        root_path = os.getcwd()
    savePath = os.path.join(root_path, 'scalar')
    # if not os.path.exists(savePath):
    #     os.mkdir(savePath)
    if 'epoch' in root_path:
        root_path = os.path.split(root_path)[0]
        savePath = os.path.join(root_path, 'scalar')
    print()
    print('-' * 80)
    print('Note: Scalar restored from %s' % savePath)
    for k in args:
        if k != 'root_path':
            f = open(os.path.join(savePath, '%s' % k), 'rb')
            # eval('%s = pickle.load(f)' % k)
            yield eval('pickle.load(f)')
            f.close()


def getMises(sig):
    mises = np.sqrt(sig[..., 0] ** 2 -
                    sig[..., 0] * sig[..., 1] +
                    sig[..., 1] ** 2 +
                    3. * sig[..., 2] ** 2)
    return mises


def dMisesdSig(sig):
    mises = getMises(sig)
    sqr3 = np.sqrt(3.)
    if mises == 0:
        dfds = np.array([1., 1., sqr3])
    else:
        dfds = np.array([(2. * sig[..., 0] - sig[..., 1]) / 2. / mises,
                         (2. * sig[..., 1] - sig[..., 0]) / 2. / mises,
                         3. * sig[..., 2] / mises])
    return dfds


def findDevice(useGPU=True):
    print()
    print('-' * 80)
    if useGPU and torch.cuda.is_available():
        device = torch.device('cuda')
        print('\t%s is used in this calculation' % torch.cuda.get_device_name(0))
    else:
        device = torch.device('cpu')
        print('\tOnly CPU is used in this calculation')
    return device


class modelTrainning:
    def __init__(self, model, savedPath, patienceNum=10, epochMax=100000, normalizationFlag=True, batchSize=1024, dyWeight=1.0):
        self.model = model
        self.savedPath = savedPath
        self.patienceNum = patienceNum
        self.epochMax = epochMax
        self.normalizationFlag = normalizationFlag
        self.batchSize = batchSize
        self.lossCalculator = tf.keras.losses.MeanSquaredError()
        self.epoch_loss_avg = tf.keras.metrics.Mean()
        self.epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        self.dyWeight = dyWeight
        self.verboseInterval = 10
        # if 'adam' in optimizerSTR:
        #     self.optimizer = tf.keras.optimizers.Adam()
        # elif 'sgd' in optimizerSTR:
        #     self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.01) # Loss is easily yo be nan with LBFGS optimizer
        #     self.verboseInterval = 10
        # else:
        #     raise ValueError('Please input a right keyword for optimizer selection! (%s) ' % optimizerSTR)

        self.x_mean, self.x_std, self.y_mean, self.y_std, self.dy_mean, self.dy_std, \
        self.x_min, self.x_max, self.y_min, self.y_max, self.dy_min, self.dy_max = \
            pickle_load('x_mean', 'x_std', 'y_mean', 'y_std', 'dy_mean', 'dy_std',
                        'x_min', 'x_max', 'y_min', 'y_max', 'dy_min', 'dy_max', root_path=self.savedPath)

    def lossFunction(self, x, y, dy, training):
        y_pred, dy_pred = self.model.gradient(x, training=training)
        return self.lossCalculator(y, y_pred)+self.dyWeight*self.lossCalculator(dy, dy_pred)

    def grad(self, x, y, dy, training=True):
        # Create here your gradient and optimizor
        with tf.GradientTape() as tape:
            loss_value = self.lossFunction(x, y, dy, training=training)
        return loss_value, tape.gradient(loss_value, self.model.trainable_variables)

    def trainModel(self, x, y, dy):
        startTime = time.time()
        # network training
        print()
        print('=' * 80)
        print('\tTraining ....')
        print()

        self.echo(samplesLen=len(x))

        # data normalization
        if self.normalizationFlag == True:
            x_normed = tf.constant(minMaxLinear(x, xmin=self.x_min, xmax=self.x_max), dtype=tf.dtypes.float32)
            y_normed = tf.constant(minMaxLinear(y, xmin=self.y_min, xmax=self.y_max), dtype=tf.dtypes.float32)
            dy_normed = tf.constant(minMaxLinear(dy, xmin=self.dy_min, xmax=self.dy_max), dtype=tf.dtypes.float32)
        else:
            x_normed = tf.constant(x, dtype=tf.dtypes.float32)
            y_normed = tf.constant(y, dtype=tf.dtypes.float32)
            dy_normed = tf.constant(dy, dtype=tf.dtypes.float32)

        epoch = 0
        minLoss, trialNum = 1e32, 0

        # End epoch
        train_loss_results = []
        train_accuracy_results = []

        while True:
            if self.batchSize == 0:
                self.trainOnce(x=x_normed, y=y_normed, dy=dy_normed)
            else:
                for i in range(0, len(x_normed)-self.batchSize, self.batchSize):
                    x_normed_temp = x_normed[i:i+self.batchSize]
                    y_normed_temp = y_normed[i:i + self.batchSize]
                    dy_normed_temp = dy_normed[i:i+self.batchSize]

                    self.trainOnce(x=x_normed_temp, y=y_normed_temp, dy=dy_normed_temp)

            # if epoch % self.verboseInterval == 0 and epoch != 0:
            lossEvaluation = self.lossFunction(x_normed, y_normed, dy_normed, training=False)
            # Track progress
            self.epoch_loss_avg.update_state(lossEvaluation)  # Add current batch loss
            loss = self.epoch_loss_avg.result()
            if loss < minLoss:
                trialNum = 0
                minLoss = loss
                message = 'Improved!'
                saveDir = os.path.join(self.savedPath, 'epoch_%d' % epoch)
                try:
                    os.mkdir(saveDir)
                except:
                    pass
                # self.model.fit(x_normed, y_normed, dy_normed)
                self.model.save(os.path.join(saveDir, 'entire_model'))
            else:
                trialNum += 1
                message = 'Noimproved! in %d tirals' % (trialNum * self.verboseInterval)

            info = "Epoch: %d \t Error: %.3e \t %s \ttimeConsumed: %.3e mins\n" % \
                   (epoch, loss, message, (time.time() - startTime) / 60.)
            print(info)
            writeDown(info, self.savedPath, appendFlag=True)

            if epoch >= self.epochMax:
                info = '\n' + '-' * 80 + '\n' + \
                       'Training Ended till epoch: %d >= epochMax %d' % (epoch, self.epochMax) + '\n' + \
                       "Epoch: %d \t Error: %.3e \t %s" % (epoch, loss, message) + '\n' + \
                       'timeConsumed: %.3f mins ' % ((time.time() - startTime) / 60.)
                print(info)
                writeDown(info, self.savedPath, appendFlag=True)
                break
            if trialNum > self.patienceNum:
                info = '\n' + '-' * 80 + '\n' + \
                       'Training Ended till trialNum: %d >= patienceNum %d' % (trialNum, self.patienceNum) + '\n' + \
                       "Epoch: %d \t Error: %.3e \t %s" % (epoch, loss, message) + '\n' + \
                       'timeConsumed: %.3f mins ' % ((time.time() - startTime) / 60.)
                print(info)
                writeDown(info, self.savedPath, appendFlag=True)
                break
            epoch += self.verboseInterval
        self.model.save(os.path.join(self.savedPath, 'entire_model'))

    def echo(self, samplesLen, appendFlag=False):
        info = '-' * 80 + '\n' + \
               '%s' % time.strftime('%Y/%m/%d  %H:%M:%S') + '\n' + \
               'PatienceNum:\t%d' % self.patienceNum + '\n' + \
               'Save path:\t%s' % self.savedPath + '\n' + \
               'Model architecture:\t %s' % self.model + '\n' + \
               'Number of training samples:\t%d' % samplesLen + '\n' + \
               '-' * 80 + '\n'
        print(info)
        if not appendFlag:
            writeDown(info, savePath=self.savedPath, appendFlag=False)
        return

    def trainOnce(self, x, y, dy):
        # Optimize the model
        self.model.fit(x=x, y=[y, dy], epochs=self.verboseInterval)


class Restore:
    def __init__(self, savedPath, device, normalizationFlag=True):
        self.device = device
        self.savedPath = savedPath
        self.normalizationFlag = normalizationFlag
        self.model = tf.keras.models.load_model(os.path.join(savedPath, 'entire_model'))
        print('\n'+'-'*80+'\n'+'Model restored from %s' % os.path.join(savedPath, 'entire_model'))
        self.x_mean, self.x_std, self.y_mean, self.y_std, self.dy_mean, self.dy_std, \
        self.x_min, self.x_max, self.y_min, self.y_max, self.dy_min, self.dy_max = \
            pickle_load('x_mean', 'x_std', 'y_mean', 'y_std', 'dy_mean', 'dy_std',
                        'x_min', 'x_max', 'y_min', 'y_max', 'dy_min', 'dy_max', root_path=self.savedPath)

    def evaluation(self, x, y, dy):
        y_origin, dy_origin = self.prediction(x)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(y.reshape(-1), y_origin.reshape(-1))
        plt.tight_layout()
        plt.savefig(os.path.join(self.savedPath, 'Prediction.png'), dpi=200)
        plt.close()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(dy.reshape(-1), dy_origin.reshape(-1))
        plt.tight_layout()
        plt.savefig(os.path.join(self.savedPath, 'dPrediction.png'), dpi=200)
        plt.close()

    def prediction(self, x):
        if self.normalizationFlag:
            x_normed = tf.constant(minMaxLinear(x, xmin=self.x_min, xmax=self.x_max))
        else:
            x_normed = tf.constant(x, dtype=tf.dtypes.float32)

        y, dy = self.model.gradient(x_normed)
        if self.normalizationFlag:
            y_origin, dy_origin = self.cast2origin(y, dy)
        else:
            y_origin, dy_origin = y.numpy(), dy.numpy()
        return y_origin, dy_origin

    def cast2origin(self, y, dy):
        # y_origin = normalizeReverse(x=y.cpu().detach().numpy(), xmean=self.y_mean, xstd=self.y_std)
        # dy_origin = normalizeReverse(x=dy.cpu().detach().numpy(), xmean=self.dy_mean, xstd=self.dy_std)
        y_origin = minMaxLinearReversed(x=y.numpy(), xmin=self.y_min, xmax=self.y_max)
        dy_origin = minMaxLinearReversed(x=dy.numpy(), xmin=self.dy_min, xmax=self.dy_max)
        return y_origin, dy_origin


def writeDown(info, savePath, appendFlag=False):
    if appendFlag:
        # print('-'*100)
        # print('Append the history file and retain!')
        f = open(os.path.join(savePath, 'history.dat'), 'a')
    else:
        # print('-'*100)
        # print('Delete the history file and begin training!')
        f = open(os.path.join(savePath, 'history.dat'), 'w')
    f.writelines(info)
    f.close()


def SigGeneration(root_path):
    sig = np.random.random(size=(1000, 3))
    mises = getMises(sig).reshape(-1, 1)
    dmisesdsig = np.array([dMisesdSig(i) for i in sig]).reshape(-1, 3)

    saveScalar(x=sig, y=mises, dy=dmisesdsig, root_path=root_path)
    return sig, mises, dmisesdsig


def dataReader(root_path, filePath = './MCCData/results', readContent='sig'):
    '''
        Read dataset from the random loading results based on mises model and isotropic hardening
    :param filePath:
    :param root_path:
    :return:
    '''
    data = np.empty(shape=(1, 14))
    for i in os.listdir(filePath):
        if '.dat' in i:
            data = np.concatenate((data, np.loadtxt(os.path.join(filePath, i), delimiter=',')), axis=0)
    data = data[1:]
    sig = data[:, :3]
    # mises = data[:, 6:7]
    mises = getMises(sig).reshape(-1, 1)
    epsPlastic = data[:, 7:8]
    H = getH(epsPlastic).reshape(-1, 1)
    dHdEps = get_dHdEps(epsPlastic).reshape(-1, 1)
    dmisesdsig = np.array([dMisesdSig(i) for i in sig]).reshape(-1, 3)
    if 'sig' in readContent:
        saveScalar(x=sig, y=mises, dy=dmisesdsig, root_path=root_path)
        return sig, mises, dmisesdsig
    else:
        saveScalar(x=epsPlastic, y=H, dy=dHdEps, root_path=root_path)
        return epsPlastic, H, dHdEps


if __name__ == "__main__":
    layerList = 'dmdd'
    savedPath = os.path.join('misesModel', layerList)
    if not os.path.exists(savedPath):
        os.mkdir(savedPath)

    device = findDevice()

    sig, mises, dmisesdsig = dataReader(root_path=savedPath)
    # sig, mises, dmisesdsig = SigGeneration(root_path=savedPath)

    sampleNum = 1000
    index_random = np.random.permutation(range(len(sig)))
    sig, mises, dmisesdsig = sig[index_random[:sampleNum]], \
                             mises[index_random[:sampleNum]], dmisesdsig[index_random[:sampleNum]]

    # ------------------------------------------------
    # model training
    misesNet = TensorFlowModel(outputNum=1)
    misesNet.compile(optimizer='adam', loss='mse')
    _ = misesNet(tf.constant(sig))
    misesNet.summary()
    mTrain = modelTrainning(model=misesNet, savedPath=savedPath,
                            normalizationFlag=True,
                            epochMax=100, batchSize=0, dyWeight=0.1,)
    mTrain.trainModel(x=sig, y=mises, dy=dmisesdsig)

    # restore
    # model_evaluation = Restore(savedPath=os.path.join(savedPath, 'epoch_5000'), device=device, normalizationFlag=True)
    model_evaluation = Restore(savedPath=savedPath, device=device, normalizationFlag=True)
    model_evaluation.evaluation(x=sig, y=mises, dy=dmisesdsig)
    print()
