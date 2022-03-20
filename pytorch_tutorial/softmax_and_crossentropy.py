import torch 
import torch.nn as nn
import numpy as np

# softmax
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print('softmax numpy:', outputs)

# torch ver.

x = torch.tensor([2.0, 1.0 ,0.1])
outputs = torch.softmax(x, dim=0)
print(outputs)

# crossentropy
def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss

Y = np.array([1, 0, 0])

y_pred_good = np.array([0.7, 0.2, 0.1])
y_pred_bad = np.array([0.1, 0.3, 0.6])

l1 = cross_entropy(Y, y_pred_good)
l2 = cross_entropy(Y, y_pred_bad)

print(f'l1 : {l1:.4f}')
print(f'l2 : {l2:.4f}')

# torch ver.

loss = nn.CrossEntropyLoss()

Y = torch.tensor([0])
y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])
y_pred_bad = torch.tensor([[0.5, 2.0, 0.3]])

l1 = loss(y_pred_good, Y)
l2 = loss(y_pred_bad, Y)

print(l1.item(), l2.item())

_, pred1 = torch.max(y_pred_good, 1)
_, pred2 = torch.max(y_pred_bad, 1)

print(pred1, pred2)

# multiclass clasification?
class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet2, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.relu = nn.Relu()
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.linear(x)
        out = self.relu(out)
        out = self.linear2(out)

        return torch.sigmoid(out)

model = NeuralNet2(input_size=28*28, hidden_size=5)
criterion = nn.BCELoss()