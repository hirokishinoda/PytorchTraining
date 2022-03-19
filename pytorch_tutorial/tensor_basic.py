from os import device_encoding
import torch
import numpy as np

# 複数次元のテンソル（行列）を作成可能
x = torch.empty(2, 3)
print(x)

# 要素が1のテンソルと型の指定
x = torch.ones(2, 2, dtype=torch.double)
print(x)
print(x.dtype)

# リストからテンソルへの変換
x = torch.tensor([2.5, 0.1])
print(x)
print(x.size())

# 足し算
print("-- add --")
x = torch.rand(2, 2)
y = torch.rand(2, 2)
print(x)
print(y)
z = x + y
# z = torch.add(x, y) # 同じ操作
# y.add_(x) # yにxを足す操作
print(z)

# 引き算
print("-- sub --")
z = x - y
# x = torch.sub(x, y)
# x.sub_(y)
print(z)

# 掛け算
print("-- mul --")
z = x * y
# z = torch.mul(x, y)
# x.mul_(y)
print(z)

# 割り算も同じ

# スライス
print("-- slice --")
x = torch.rand(5, 3)
print(x)
print(x[:, 2])
print(x[2, 2].item())

# view
print("-- view --")
x = torch.rand(4, 4)
print(x)
y = x.view(16)
print(y)
y = x.view(-1, 8)
print(y)

# convert to numpy
print("-- convert to numpy --")
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
a.add_(1)
# メモリは共有されてるが型が違う
print(a)
print(b)

# 
print("-- convert to torch --")
a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)

# using GPU
print("-- using GPU --")
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device)
    y = torch.ones(5)
    y = y.to(device)
    z = x + y
    # z.numpy() # Error numpy use only on cpu
    z = z.to("cpu")

# Grad True or False
print("-- Grad True or False --")
x = torch.ones(5, requires_grad=True)
print(x)