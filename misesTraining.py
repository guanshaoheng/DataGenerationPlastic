import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
from torch import nn
import pickle
import torch.nn.functional as F
import copy


class SSP(nn.Softplus):
    def __init__(self, beta=1, threshold=20):
        super(SSP, self).__init__(beta, threshold)
    def forward(self, input):
        sp0 = F.softplus(torch.zeros(1), self.beta, self.threshold).item()
        return F.softplus(input, self.beta, self.threshold) - sp0


class Net(nn.Module):
    def __init__(self, xmin, xmax, ymin, ymax, device,
                 activation=SSP(), inputNum=4, outputNum=1, layerList='dddmd', node=100):
        super(Net, self).__init__()
        self.inputNum = inputNum
        self.outputNum = outputNum
        self.node = node
        self.layerList = layerList
        self.layers = torch.nn.ModuleList(self.getInitLayers())
        self.activation = activation
        self.xmin, self.xmax, self.ymin, self.ymax = torch.tensor(xmin, dtype=torch.float, device=device), \
                                                     torch.tensor(xmax, dtype=torch.float, device=device), \
                                                     torch.tensor(ymin, dtype=torch.float, device=device), \
                                                     torch.tensor(ymax, dtype=torch.float, device=device)

    def normalization(self, x, xmin, xmax, reverse=False):
        if reverse:
            normed = (xmax-xmin)*x+xmin
        else:
            normed = (x-xmin)/(xmax-xmin)
        return normed

    def getInitLayers(self, ):
        layers = []
        num_layers = len(self.layerList)
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(self.inputNum, self.node))
            elif i == num_layers - 1:
                layers.append(nn.Linear(self.node, self.outputNum))
            else:
                layers.append(nn.Linear(self.node, self.node))
        return layers

    def forward(self, x):
        # normalization
        x = self.normalization(x, self.xmin, self.xmax)
        num_layers = len(self.layers)
        for i, key in enumerate(self.layerList[:-1]):
            if key == 'd':
                x = self.activation(self.layers[i](x))
            else:
                x = self.activation(self.layers[i](x * x))
        x = self.layers[num_layers - 1](x)
        # reverse normalization
        y = self.normalization(x, self.ymin, self.ymax, reverse=False)
        return y


class NetContrained(nn.Module):
    def __init__(self, activation=nn.ReLU(), inputNum=3, outputNum=1, layerList='dddd', node=100):
        super(NetContrained, self).__init__()
        self.inputNum = inputNum
        self.outputNum = outputNum
        self.node = node
        self.activation = activation
        self.layerList = layerList
        self.layers = torch.nn.ModuleList(self.getInitLayers())

    def getInitLayers(self, ):
        layers = []
        num_layers = len(self.layerList)
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(9, self.node))
            elif i == num_layers - 1:
                layers.append(nn.Linear(self.node, self.outputNum))
            else:
                layers.append(nn.Linear(self.node, self.node))
        return layers

    def forward(self, x):
        x2 = x*x
        x = torch.concat((x, x2, x[:, 0:1]*x[:, 1:2], x[:, 1:2]*x[:, 2:3], x[:, 0:1]*x[:, 2:3],), axis=1)
        num_layers = len(self.layers)
        for i, key in enumerate(self.layerList[:-1]):
            if key == 'd':
                x = self.activation(self.layers[i](x))
            else:
                x = self.activation(self.layers[i](x * x))
        x = self.layers[num_layers - 1](x)
        return x

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


def dMises2dSig(sig):
    dfds = np.array([(2. * sig[..., 0] - sig[..., 1]),
                     (2. * sig[..., 1] - sig[..., 0]),
                     6. * sig[..., 2]])
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
    def __init__(self, model, savedPath, device, optimizerSTR='adam',
                 patienceNum=20, epochMax=100000, normalizationFlag=True, batchSize=1024, dyWeight=1.0, num_batches=10):
        self.model = model
        self.savedPath = savedPath
        self.device = device
        self.patienceNum = patienceNum
        self.epochMax = epochMax
        self.normalizationFlag = normalizationFlag
        self.batchSize = batchSize
        self.lossCalculator = torch.nn.MSELoss()
        self.dyWeight = dyWeight
        self.state = copy.deepcopy(self.model.state_dict())
        self.num_batches = 10
        if 'adam' in optimizerSTR:
            # self.optimizer = torch.optim.Adam(self.model.parameters(), )
            self.optimizer = torch.optim.NAdam(self.model.parameters(), )
            self.verboseInterval = 10
        elif 'lbfgs' in optimizerSTR:
            self.optimizer = torch.optim.LBFGS(
                self.model.parameters(), )  # Loss is easily yo be nan with LBFGS optimizer
            self.verboseInterval = 5
        else:
            raise ValueError('Please input a right keyword for optimizer selection! (%s) ' % optimizerSTR)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2000, gamma=0.95)
        # self.x_mean, self.x_std, self.y_mean, self.y_std, self.dy_mean, self.dy_std, \
        # self.x_min, self.x_max, self.y_min, self.y_max, self.dy_min, self.dy_max = \
        #     pickle_load('x_mean', 'x_std', 'y_mean', 'y_std', 'dy_mean', 'dy_std',
        #                 'x_min', 'x_max', 'y_min', 'y_max', 'dy_min', 'dy_max', root_path=self.savedPath)
        # self.x_min = torch.tensor(self.x_min, dtype=torch.float, device=self.device)
        # self.x_max = torch.tensor(self.x_max, dtype=torch.float, device=self.device)
        # self.y_min = torch.tensor(self.y_min, dtype=torch.float, device=self.device)
        # self.y_max = torch.tensor(self.y_max, dtype=torch.float, device=self.device)
        # self.dy_min = torch.tensor(self.dy_min, dtype=torch.float, device=self.device)
        # self.dy_max = torch.tensor(self.dy_max, dtype=torch.float, device=self.device)

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
            x_normed = minMaxLinear(x, xmin=self.x_min, xmax=self.x_max)
            y_normed = minMaxLinear(y, xmin=self.y_min, xmax=self.y_max)
            dy_normed = minMaxLinear(dy, xmin=self.dy_min, xmax=self.dy_max)
            # x_normed = torch.tensor(minMaxLinear(x, xmin=self.x_min, xmax=self.x_max),
            #                         dtype=torch.float, requires_grad=True).to(self.device)
            # y_normed = torch.tensor(minMaxLinear(y, xmin=self.y_min, xmax=self.y_max),
            #                         dtype=torch.float, requires_grad=True).to(self.device)
            # dy_normed = torch.tensor(minMaxLinear(dy, xmin=self.dy_min, xmax=self.dy_max),
            #                          dtype=torch.float, requires_grad=True).to(self.device)
            # x_normed = torch.tensor(normalize(x, xmean=self.x_mean, xstd=self.x_std),
            #                         dtype=torch.float, requires_grad=True).to(self.device)
            # y_normed = torch.tensor(normalize(y, xmean=self.y_mean, xstd=self.y_std),
            #                         dtype=torch.float, requires_grad=True).to(self.device)
            # dy_normed = torch.tensor(normalize(dy, xmean=self.dy_mean, xstd=self.dy_std),
            #                          dtype=torch.float, requires_grad=True).to(self.device)
        else:
            x_normed = x
            y_normed = y
            dy_normed = dy

        epoch = 0
        minLoss, trialNum = 1e32, 0
        while True:
            self.scheduler.step()  # used for the learning rate decay
            indices = np.arange(len(x_normed))
            np.random.shuffle(indices)
            loss0, loss1 = 0., 0.
            for batch in np.split(indices, self.num_batches):
                x_normed_temp = torch.tensor(x_normed[batch], dtype=torch.float32, device=self.device)
                y_normed_temp = torch.tensor(y_normed[batch], dtype=torch.float32, device=self.device)
                dy_normed_temp = torch.tensor(dy_normed[batch], dtype=torch.float32, device=self.device)
                x_normed_temp.requires_grad = True
                loss0_batch, loss1_batch = self.trainOnce(x=x_normed_temp, y=y_normed_temp, dy=dy_normed_temp)
                loss0 += loss0_batch*len(batch)
                loss1 += loss1_batch*len(batch)

            if epoch % self.verboseInterval == 0:
                loss = loss0+self.dyWeight*loss1
                if loss < minLoss:
                    trialNum = 0
                    minLoss = loss
                    message = 'Improved!'
                    self.state = copy.deepcopy(self.model.state_dict())
                else:
                    trialNum += 1
                    message = 'Noimproved! in %d tirals' % (trialNum * self.verboseInterval)

                info = "Epoch: %d \t lr: %.3e \t Loss: %.3e \t Loss0: %.3e \t Loss1: %.3e \t %s \ttimeConsumed: %.3e mins\n" % \
                       (epoch, self.optimizer.param_groups[0]['lr'], loss, loss0, loss1, message, (time.time() - startTime) / 60.)
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
            if (epoch+1) % (self.verboseInterval*500) == 0:
                saveDir = os.path.join(self.savedPath, 'epoch_%d' % (epoch+1))
                try:
                    os.mkdir(saveDir)
                except:
                    pass
                torch.save(self.model, os.path.join(saveDir, 'entire_model.pt'))
            epoch += 1
        self.model.load_state_dict(self.state)
        torch.save(self.model, os.path.join(self.savedPath, 'entire_model.pt'))

    def echo(self, samplesLen, appendFlag=False):
        info = '-' * 80 + '\n' + \
               '%s' % time.strftime('%Y/%m/%d  %H:%M:%S') + '\n' + \
               'PatienceNum:\t%d' % self.patienceNum + '\n' + \
               'Save path:\t%s' % self.savedPath + '\n' + \
               'Model architecture:\t %s' % self.model + '\n' + \
               'Optimizer:\t%s' % self.optimizer + '\n' + \
               'Number of training samples:\t%d' % samplesLen + '\n' + \
               '-' * 80 + '\n'
        print(info)
        if not appendFlag:
            writeDown(info, savePath=self.savedPath, appendFlag=False)
        return

    def trainOnce(self, x, y, dy):
        self.optimizer.zero_grad()
        yPrediction = self.model(x)
        dyPrediction_dx = torch.autograd.grad(outputs=yPrediction,
                                              inputs=x,
                                              grad_outputs=torch.ones_like(yPrediction).to(self.device),
                                              retain_graph=True,
                                              create_graph=True)[0]
        # norm = torch.linalg.norm(dyPrediction_dx, dim=1)
        loss0 = self.lossCalculator(yPrediction, y)
        loss1 = self.lossCalculator(dyPrediction_dx, dy)
        loss = loss0+self.dyWeight*loss1
        loss.backward()
        self.optimizer.step()
        return loss0.item(), loss1.item()


class Restore:
    def __init__(self, savedPath, device, normalizationFlag=True):
        self.device = device
        self.savedPath = savedPath
        self.normalizationFlag = normalizationFlag
        self.model = torch.load(os.path.join(savedPath, 'entire_model.pt')).to(device)
        # self.x_mean, self.x_std, self.y_mean, self.y_std, self.dy_mean, self.dy_std, \
        # self.x_min, self.x_max, self.y_min, self.y_max, self.dy_min, self.dy_max = \
        #     pickle_load('x_mean', 'x_std', 'y_mean', 'y_std', 'dy_mean', 'dy_std',
        #                 'x_min', 'x_max', 'y_min', 'y_max', 'dy_min', 'dy_max', root_path=self.savedPath)

    def evaluation(self, x, y, dy):
        y_origin, dy_origin = self.prediction(x)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(y.reshape(-1), y_origin.reshape(-1))
        plt.tight_layout()
        plt.axis('equal')
        plt.savefig(os.path.join(self.savedPath, 'Prediction.png'), dpi=200)
        plt.close()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(dy.reshape(-1), dy_origin.reshape(-1))
        plt.tight_layout()
        plt.axis('equal')
        plt.savefig(os.path.join(self.savedPath, 'dPrediction.png'), dpi=200)
        plt.close()

    def prediction(self, x):
        if self.normalizationFlag:
            # x_normed = torch.tensor(normalize(x, xmean=self.x_mean, xstd=self.x_std),
            #                         dtype=torch.float, requires_grad=True).to(self.device)
            x_normed = torch.tensor(minMaxLinear(x, xmin=self.x_min, xmax=self.x_max),
                                    dtype=torch.float, requires_grad=True).to(self.device)
        else:
            x_normed = torch.tensor(x, dtype=torch.float, requires_grad=True).to(self.device)

        y = self.model(x_normed)

        dy = torch.autograd.grad(outputs=y, inputs=x_normed,
                                 grad_outputs=torch.ones(y.size()).to(self.device),
                                 retain_graph=True,
                                 create_graph=True, only_inputs=True)[0]
        if self.normalizationFlag:
            y_origin, dy_origin = self.cast2origin(y, dy)
        else:
            y_origin, dy_origin = y.cpu().detach().numpy(), dy.cpu().detach().numpy()
        return y_origin, dy_origin

    def cast2origin(self, y, dy):
        # y_origin = normalizeReverse(x=y.cpu().detach().numpy(), xmean=self.y_mean, xstd=self.y_std)
        # dy_origin = normalizeReverse(x=dy.cpu().detach().numpy(), xmean=self.dy_mean, xstd=self.dy_std)
        y_origin = minMaxLinearReversed(x=y.cpu().detach().numpy(), xmin=self.y_min, xmax=self.y_max)
        dy_origin = minMaxLinearReversed(x=dy.cpu().detach().numpy(), xmin=self.dy_min, xmax=self.dy_max)
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
    # mises2 = mises*mises
    epsPlastic = data[:, 7:8]
    H = getH(epsPlastic).reshape(-1, 1)
    dHdEps = get_dHdEps(epsPlastic).reshape(-1, 1)
    dmisesdsig = np.array([dMisesdSig(i) for i in sig]).reshape(-1, 3)
    # dmises2dsig = np.array([dMises2dSig(i) for i in sig]).reshape(-1, 3)
    if 'sig' in readContent:
        saveScalar(x=sig, y=mises, dy=dmisesdsig, root_path=root_path)
        # return sig, mises, dmisesdsig, mises2, dmises2dsig
        return sig, mises, dmisesdsig
    else:
        saveScalar(x=epsPlastic, y=H, dy=dHdEps, root_path=root_path)
        return epsPlastic, H, dHdEps


if __name__ == "__main__":
    layerList = 'dmdmd'
    savedPath = os.path.join('misesModel', layerList)
    functionGeneration = False
    normalizationFlag = False
    savedPath += ('_generation' if functionGeneration else '_data')
    savedPath += ('_scaled' if normalizationFlag else '_Noscaled')
    if not os.path.exists(savedPath):
        os.mkdir(savedPath)
    if functionGeneration:
        sig, mises, dmisesdsig = SigGeneration(root_path=savedPath)
    else:
        sig, mises, dmisesdsig = dataReader(root_path=savedPath)


    sampleNum = 5000
    index_random = np.random.permutation(range(len(sig)))
    sig, mises, dmisesdsig = sig[index_random[:sampleNum]], \
                             mises[index_random[:sampleNum]], dmisesdsig[index_random[:sampleNum]]

    # ------------------------------------------------
    # model training
    device = findDevice()
    x_min, x_max, y_min, y_max, dy_min, dy_max = \
        pickle_load('x_min', 'x_max', 'y_min', 'y_max', 'dy_min', 'dy_max', root_path=savedPath)
    misesNet = Net(xmin=x_min, xmax=x_max, ymin=y_min, ymax=y_max, device=device,
                   activation=SSP(), inputNum=3, outputNum=1, layerList=layerList, node=20).to(device=device)
    mTrain = modelTrainning(model=misesNet, savedPath=savedPath,
                            device=device, optimizerSTR='adam',
                            normalizationFlag=normalizationFlag,
                            epochMax=int(1e6), batchSize=0, dyWeight=4e8, patienceNum=50)
    mTrain.trainModel(x=sig, y=mises, dy=dmisesdsig)

    # restore
    # model_evaluation = Restore(savedPath=os.path.join(savedPath, 'epoch_149999'),
    #                            device=device, normalizationFlag=normalizationFlag)
    model_evaluation = Restore(savedPath=savedPath, device=device, normalizationFlag=normalizationFlag)
    model_evaluation.evaluation(x=sig, y=mises, dy=dmisesdsig)
    print()
