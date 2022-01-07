import torch

device = 'cuda'
a = torch.randn(3, dtype=torch.float).to(device)
b = torch.randn(3, dtype=torch.float).to(device)

optimizer = torch.optim.Adam(params=[a, b], )

# Post facto set gradients
a.requires_grad_()
b.requires_grad_()

print("a is ", a)
print("b is ", b)

loss1 = a.sum()

loss2 = b.sum()

loss = loss1+loss2
optimizer.zero_grad()
loss.backward()

print("Gradient wrt to a is ", a.grad)
print("Gradient wrt to b is ", b.grad)
