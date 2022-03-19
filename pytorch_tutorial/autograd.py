import torch

# auto gradについて
x = torch.randn(3, requires_grad=True) # requirre_grad is to enable autograd
print(x)

y = x+2
print(y)
z = (y**2)*2
print(z)
#z = z.mean()
#print(z)

#z.backward()
#print(x.grad)

v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
z.backward(v)
print(z)

# 勾配計算をしない場合
# x.requires_grad_(False)
# x.detach()
# with.torch.no_grad():

x = torch.randn(3, requires_grad=True) # requirre_grad is to enable autograd
print(x)

x.requires_grad_(False)
print(x)

# 仮の訓練コード
weights = torch.ones(4, requires_grad=True)

optimizer = torch.optim.SGD(weights, lr=0.01)
optimizer.step()
optimizer.zero_grad()

for epoch in range(10):
    model_output = (weights*3).sum()

    model_output.backward()

    print(weights.grad)

    weights.grad.zero_() # gradが加算されるので毎回初期化するべき

