import torch
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
        self.D = self.tangentAssemble(
            lam=self.youngsModulus * self.poisson / (1 + self.poisson) / (1 - 2 * self.poisson),
            G=self.youngsModulus / 2 / (1 + self.poisson))

        # ----------------------------------
        # net initialization
        self.yieldFunction = Net(inputNum=4, outputNum=1, layerList='ddmd')

        # ----------------------------------
        # state
        self.sigma = 0.
        self.epsPlasticVector = np.zeros(3, dtype=float)
        self.epsPlastic = np.linalg.norm(self.epsPlasticVector)
        self.hardening = self.getHardening(epsPlastic=0.)

    def diffOFyieldFunction(self, x):  # x should be a 1-d torch.FloatTensor
        SingedDistance = self.yieldFunction(x)
        temp = torch.autograd.grad(outputs=SingedDistance, inputs=x, retain_graph=True, create_graph=True,
                                   grad_outputs=SingedDistance.size())
        dfds = temp[:, :3]
        dfdep = temp[:, 3:]
        return dfds, dfdep

    def getHardening(self, epsPlastic=None):
        if epsPlastic:
            hardeningValue = self.A * (self.epsilon0 + abs(epsPlastic)) ** self.n
        else:
            hardeningValue = self.A * (self.epsilon0 + self.epsPlastic) ** self.n
        return hardeningValue

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
        trialStress = self.sigma + dsig
        yieldValue = self.yieldFunction(torch.concat((trialStress, self.hardening), ))
        if yieldValue<0:  # elastic
            return trialStress
        else:  # plastic
            # bi-section to find the value on the yield function
            mid = self.bisectionDeps(dsig)
            self.plasticReturnMapping(deps=(1-mid)*deps)
        return

    def bisectionDeps(self, dsig):
        r0, mid, r1 = 0., 0.5, 1.0
        distance = self.yieldFunction(torch.concat((self.sigma+dsig*mid, self.hardening), ))
        while distance > 0. or distance < -10:
            if distance > 0.:
                r1 = mid
            else:
                r0 = mid
            mid = (r0 + r1)*.5
            distance = self.yieldFunction(torch.concat((self.sigma+dsig * mid, self.hardening), ))
        self.sigma += dsig*mid
        return mid

    def plasticReturnMapping(self, deps):
        dfds, dfdep = self.diffOFyieldFunction(x=torch.concat((deps, self.hardening), ))
        h = - dfdep*torch.linalg.norm(dfds)
        H = (h + dfds @ self.D @ dfds)[0, 0]
        dLambda = (dfds @ self.D @ deps / H)[0, 0]
        deps_plastic = dLambda * dfds
        epsPlastic = self.epsPlastic + np.sqrt(2/3*deps_plastic.T @ deps_plastic)[0, 0]
        return

    def getShearStrain(self, epsilon):

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