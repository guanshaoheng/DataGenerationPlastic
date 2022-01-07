import os.path
import numpy as np
import matplotlib.pyplot as plt
from MisesIsoHardeningNetworkModel import Net
import torch
from torch.autograd.functional import jacobian
import scipy.interpolate


useGPU = True
if useGPU and torch.cuda.is_available():
    device = torch.device('cuda')
    print('%s' % torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')

net = Net(inputNum=2, outputNum=1, layerList='ddmd').to(device)
n = 201
mesh = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
x = np.concatenate((mesh[0].reshape(-1, 1), mesh[1].reshape(-1, 1)), axis=1)
value = (mesh[0]**2+mesh[1]**2-1.)/2.
dvaluedx = np.concatenate((mesh[0].reshape(-1, 1), mesh[1].reshape(-1, 1)), axis=1)
y = value.reshape(-1, 1)

plt.imshow(value, vmin=np.min(value), vmax=np.max(value),
           extent=[mesh[0].min(), mesh[0].max(), mesh[1].min(), mesh[1].max()])
plt.title('Dataset')
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join('test', '0dataset.png'), dpi=200)
plt.close()

index_random = np.random.permutation(range(len(x)))[:100000]
x, y = x[index_random], y[index_random]
x_tensor = torch.tensor(x, dtype=torch.float, requires_grad=True).to(device)
y_tensor = torch.tensor(y, dtype=torch.float, requires_grad=True).to(device)

# x_tensor.requires_grad_()
# y_tensor.requires_grad_()
# dvaluedx_tensor = torch.tensor(dvaluedx, dtype=torch.float, requires_grad=True).to(device)
ones = torch.ones([len(x_tensor)]).to(device)

def plotfigure(net, num):
    nPlotPonits = 1000
    y_predict = net(x_tensor).cpu().detach().numpy()
    y_np = y_predict.flatten()
    x_mesh, y_mesh = np.meshgrid(np.linspace(-1, 1, 101), np.linspace(-1, 1, 101))
    rbf = scipy.interpolate.Rbf(x[:nPlotPonits, 0], x[:nPlotPonits, 1], y_np[:nPlotPonits], function='linear')
    value_predict = rbf(x_mesh, y_mesh)
    # residual = y_predict-y

    plt.imshow(value_predict, vmin=np.min(value_predict), vmax=np.max(value_predict), origin='lower',
               extent=[mesh[0].min(), mesh[0].max(), mesh[1].min(), mesh[1].max()])
    infor= 'ML prediction %d' % num
    plt.title(infor)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join('test', infor+'.png'), dpi=200)
    plt.close()


# network training
# optimizer = torch.optim.Adam(net.parameters(), )
optimizer = torch.optim.LBFGS(net.parameters(), )
lossCalculator = torch.nn.MSELoss()
epochTrain = 200
# x_tensor, y_tensor = x_tensor[index_random], y_tensor[index_random]
for epoch in range(epochTrain):
    def closure():
        optimizer.zero_grad()
        yPrediction = net(x_tensor)
        dyPrediction_dx = torch.autograd.grad(outputs=yPrediction,
                                              inputs=x_tensor,
                                              grad_outputs=torch.ones(yPrediction.size()).to(device),
                                              retain_graph=True,
                                              create_graph=True,
                                              only_inputs=True)[0]
        # two_order_gradients = torch.autograd.grad(outputs=dyPrediction_dx, inputs=x_temp,
        #                                           grad_outputs=torch.ones(dyPrediction_dx.size()),
        #                                           only_inputs=True)[0]
        # loss = torch.square(torch.linalg.norm(yPrediction-y_temp)) + \
        #        torch.square(torch.linalg.norm(dyPrediction_dx) - 1.)
        # loss = torch.square(dyPrediction_dx[:, 0]) + torch.square(dyPrediction_dx[:, 1])-torch.ones(size=(len(dyPrediction_dx), ))
        # loss = lossCalculator(dyPrediction_dx[:, 0:1]**2+dyPrediction_dx[:, 1:2]**2,
        #                       torch.ones(size=(len(dyPrediction_dx), 1)))
        # loss = lossCalculator(dyPrediction_dx, x_tensor)
        norm = torch.linalg.norm(dyPrediction_dx, dim=1)
        loss = lossCalculator(norm, ones)
        loss.backward()
        return loss
    optimizer.step(closure)
    if epoch % (epochTrain//20) == 0:
        yPrediction = net(x_tensor)
        dyPrediction_dx = torch.autograd.grad(outputs=yPrediction, inputs=x_tensor,
                                              grad_outputs=torch.ones(yPrediction.size()).to(device),
                                              retain_graph=True,
                                              create_graph=True, only_inputs=True)[0]
        norm = torch.linalg.norm(dyPrediction_dx, dim=1)
        loss0 = lossCalculator(norm, ones).item()
        # loss0 = lossCalculator(dyPrediction_dx, x_tensor).item()
        # loss1 = torch.square(torch.linalg.norm(dyPrediction_dx) - 1.).item()
        print()
        print("Epoch: %d \t Error: %.3e \t" % (epoch, loss0))
        plotfigure(net, epoch)
plotfigure(net, epochTrain)


