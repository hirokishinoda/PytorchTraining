import torch

# f = w * x

# f = 2 *x
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model predict
def forward(x):
    return w * x
# loss
def loss(t, y):
    return ((y - t)**2).mean()

print(f'Predicttion before training f(5) = {forward(5):.3f}')

# Training
lr = 0.01
epochs = 20

for epoch in range(epochs):
    # predict
    y = forward(X)

    # loss
    l = loss(Y, y)

    # gradient
    l.backward()

    # update weight
    with torch.no_grad():
        w -= lr * w.grad

    # inint grads
    w.grad.zero_()

    if epoch % 1 == 0:
        print(f'epoch: {epoch+1}, w= {w:.3f}, loss= {l:.8f}')


print(f'Predicttion after training f(5) = {forward(5):.3f}')