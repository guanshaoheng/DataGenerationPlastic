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
# sigma = torch.randn(3, dtype=torch.float, requires_grad=True).to(device)

# optimizer = torch.optim.Adam(params=[a, b], )

# Post facto set gradients
# a.requires_grad_()
# b.requires_grad_()

print()
print('-'*80)
print("$\epsilon$ is \n", epsilon)
print("$\D$ is \n", D)

sigma = l2(l1(epsilon)*epsilon)

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

jacobian = torch.autograd.grad(outputs=sigma,
                               inputs=epsilon,
                               grad_outputs=torch.ones((len(sigma), 3)).to(device),
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True,
                               allow_unused=True,
                               )[0]

residual = jacobian-jacobian1-jacobian2-jacobian3
print()
print('='*80)
print('\t\t Residual:')
print(residual)
print()
print('\t We can see that the Jacobian = Jacobian1+Jacobian2+Jacobian3')
print('\t We can see that the Jacobian = Jacobian1+Jacobian2+Jacobian3')
print('\t We can see that the Jacobian = Jacobian1+Jacobian2+Jacobian3')