import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
from torch.autograd.functional import jacobian

"""
        The constitutive model is implemented via torch tensor,
        under the elastoplastic framework of Mises yield function,
        associated-flow rule, and iso-hardening.
        
        Note: all of the work is implemented in quasi-static mode.
    
        Author: Shaoheng Guan
        Email:  shaohengguan@gmail.com
    
        Reference:
            [1]
"""


class Net(nn.Module):
    def __init__(self, inputNum=4, outputNum=1, layerList='dmmd', node=30):
        super(Net, self).__init__()
        self.inputNum = inputNum
        self.outputNum = outputNum
        self.node = node
        self.layerList = layerList
        self.layers = torch.nn.ModuleList(self.getInitLayers())

    def getInitLayers(self, ):
        layers = []
        num_layers = len(self.layerList)
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(self.inputNum, self.node))
            elif i == num_layers-1:
                layers.append(nn.Linear(self.node, self.outputNum))
            else:
                layers.append(nn.Linear(self.node, self.node))
        return layers

    def forward(self, x):
        num_layers = len(self.layers)
        for i, key in enumerate(self.layerList[:-1]):
            if key == 'd':
                x = torch.relu(self.layers[i](x))
            else:
                x = torch.relu(self.layers[i](x*x))
        x = self.layers[num_layers - 1](x)
        return x


class ConstitutiveNetwork:
    def __init__(self):
        # ---------------------------------
        # parameters
        # material parameters
        self.youngsModulus = 200e9
        self.poisson = 0.3
        self.A = 500e6
        self.n = 0.2
        self.epsilon0 = 0.05
        self.yieldStress = 200e6
        self.hardening = self.getHardening(epsPlastic=0.)
        self.D = self.tangentAssemble(
            lam=self.youngsModulus * self.poisson / (1 + self.poisson) / (1 - 2 * self.poisson),
            G=self.youngsModulus / 2 / (1 + self.poisson))

        # ----------------------------------
        # net initialization
        self.yieldFunction = Net()

    def diffOFyieldFunction(self, x):  # x should be a 1-d torch.FloatTensor
        temp = jacobian(self.yieldFunction, x)
        dfds = temp[:, :3]
        dfdep = temp[:, 3:]
        return dfds, dfdep

    def getHardening(self, epsPlastic):
        hardingValue = self.A * (self.epsilon0 + abs(epsPlastic)) ** self.n
        return hardingValue

    def tangentAssemble(self, lam, G):
        D = np.zeros([3, 3])
        for i in range(2):
            for j in range(2):
                D[i, j] += lam
        D[0, 0] += 2 * G
        D[1, 1] += 2 * G
        D[2, 2] += G
        return D

    def plasticFlowFunction(self):
        return

    def getStress(self, deps):
        dsig = self.D@deps.reshape(-1, 1)
        
        return

    def getTangent(self):

        return

    def lossFunction(self):
        return


class NetTf:
    def __init__(self, inputNum=4, outputNum=1, layerList='dmmd', node=30):
        self.inputNum = inputNum
        self.outputNum = outputNum
        self.node = node
        self.layerList = layerList
        self.layers = torch.nn.ModuleList(self.getInitLayers())





if __name__ == "__main__":
    net = Net()
    x = Variable(torch.from_numpy(np.array(np.random.random(size=4))).type(torch.FloatTensor))
    net.zero_grad()
    y = net(x)
    dfds = jacobian(net, x)