import numpy as np
import matplotlib.pyplot as plt
from MisesIsoHardeningNetworkModel import Net
import torch
from torch.autograd.functional import jacobian

net = Net(inputNum=2, outputNum=1)
n = 201
mesh = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
value = mesh[0]**2+mesh[1]**2

plt.imshow(value, extent=[mesh[0].min(), mesh[0].max(), mesh[1].min(), mesh[1].max()])
plt.show()

x = np.concatenate((mesh[0].reshape(-1, 1), mesh[1].reshape(-1, 1)), axis=1)
y = x[:, 0]**2+x[:, 1]**2

x_tensor = torch.tensor(x, dtype=torch.float, requires_grad=True)
y_tensor = torch.tensor(y, dtype=torch.float, requires_grad=True)


# network training
optimizer = torch.optim.Adam(net.parameters())
lossCalculator = torch.nn.L1Loss()
epochTrain = 100
for epoch in range(epochTrain):
    index_random = np.random.permutation(range(len(x)))
    for i, index_temp in enumerate(index_random[:100]):
        x_temp = x_tensor[index_temp]
        y_temp = y_tensor[index_temp]
        def closure():
            optimizer.zero_grad()
            output = net(x_temp)
            grad = torch.autograd.grad(output, x_temp, create_graph=True)[0]
            loss = torch.square(torch.linalg.norm(grad)-1.)
            # output = net(x_temp)
            # loss = lossCalculator(output, y_temp)
            loss.backward()
            return loss
        optimizer.step(closure)

nPlotPonits = 1000
index_random = np.random.permutation(range(len(x)))
y_predict = net(x_tensor).detach()
y_np = np.array(y_predict).flatten()
plt.scatter(x[:, 0][index_random[:nPlotPonits]], x[:, 1][index_random[:nPlotPonits]],
            s=-10*y_np[index_random[:nPlotPonits]])
plt.axis('equal')
plt.show()