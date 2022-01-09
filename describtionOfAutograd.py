import torch


'''
    This script is uesed to describe the Autograd.grad in PyTorch.
    
    As you can see that the Jacobian = Jacobian1+Jacobian2+Jacobian3
'''

device = torch.device('cuda')
l1 = torch.nn.Linear(in_features=3, out_features=3, bias=True, device=device)
l2 = torch.nn.Linear(in_features=3, out_features=3, bias=True, device=device)
D = torch.tensor([[2, -1, 0], [-1, 2, 0], [0, 0, 3]], dtype=torch.float, requires_grad=True).to(device)
epsilon = torch.randn((5, 3), dtype=torch.float, requires_grad=True).to(device)
ones = torch.ones(len(epsilon), dtype=torch.float, device=device)
# sigma = torch.randn(3, dtype=torch.float, requires_grad=True).to(device)

params = list(l1.parameters()) + list(l2.parameters())
optimizer = torch.optim.Adam(params=params, )
criterion = torch.nn.MSELoss()

# Post facto set gradients
# a.requires_grad_()
# b.requires_grad_()

print()
print('-'*80)
print("$\epsilon$ is \n", epsilon)
print("$\D$ is \n", D)

print()
print('-'*80)
print('\t Traininig ...')
print()
epochMax = int(1e4)
for epoch in range(epochMax):
    sigma = l2(l1(epsilon) * epsilon)
    jacobian1 = torch.autograd.grad(outputs=sigma[:, 0:1],
                                   inputs=epsilon,
                                   grad_outputs=torch.ones((len(sigma), 1)).to(device),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True,
                                   allow_unused=True,
                                   )[0]
    jacobian2 = torch.autograd.grad(outputs=sigma[:, 1:2],
                                   inputs=epsilon,
                                   grad_outputs=torch.ones((len(sigma), 1)).to(device),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True,
                                   allow_unused=True,
                                   )[0]
    jacobian3 = torch.autograd.grad(outputs=sigma[:, 2:3],
                                   inputs=epsilon,
                                   grad_outputs=torch.ones((len(sigma), 1)).to(device),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True,
                                   allow_unused=True,
                                   )[0]

    jacobian = torch.concat((jacobian1, jacobian2, jacobian3), dim=1)

    norm = torch.linalg.norm(jacobian, dim=1)

    loss = criterion(norm, ones)
    if epoch%(epochMax//10) == 0:
        print('\tEpoch %d error: %.3e' % (epoch, loss.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

jacobian0 = torch.autograd.grad(outputs=sigma,
                                   inputs=epsilon,
                                   grad_outputs=torch.ones((len(sigma), 3)).to(device),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True,
                                   allow_unused=True,
                                   )[0]
residual = jacobian0-jacobian1-jacobian2-jacobian3

print()
print('='*80)
print('\t\t Residual:')
print(residual)
print()
print('\t We can see that the Jacobian = Jacobian1+Jacobian2+Jacobian3')
print('\t We can see that the Jacobian = Jacobian1+Jacobian2+Jacobian3')
print('\t We can see that the Jacobian = Jacobian1+Jacobian2+Jacobian3')