import numpy as np

# f = w * x

# f = 2 *x
X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)

w = 0.0

# model predict
def forward(x):
    return w * x
# loss
def loss(t, y):
    return ((y - t)**2).mean()
# gradient
# dJ/dw = 1/N 2x (w*x - y)
def gradient(x, t, y):
    return np.dot(2*x, y-t).mean()

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
    dw = gradient(X, Y, y)

    # update weight
    w -= lr * dw

    if epoch % 1 == 0:
        print(f'epoch: {epoch+1}, w= {w:.3f}, loss= {l:.8f}')


print(f'Predicttion after training f(5) = {forward(5):.3f}')