import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import gtsrb_loader
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
from tqdm import tqdm

# Hyperparamètres
lr = 3
batch_size = 32
epochs = 100


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(20, 40, 3)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(8*8*40, 128)
        self.fc2 = nn.Linear(128, 100)
        self.fc3 = nn.Linear(100, 43)

    def forward_train(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(len(x), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=True)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5, training=True)
        x = F.softmax(self.fc3(x))
        return x

    def forward_eval(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(len(x), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x


def eta(e):
    return lr/(e + 1)


def loss_fn(y_pred, y):
    return (y_pred - torch.eye(43)[y.data]).pow(2).mean()


def accuracy(images, labels):
    count = 0
    train_loader2 = DataLoader(TensorDataset(images, labels), batch_size=1000)
    for (x, y) in train_loader2:
        y_pred = model.forward_eval(x)
        count += (y_pred.max(1)[1] == y).double().data.sum()
    return 100*count/len(images)

train_images, train_labels = gtsrb_loader.train('grey', 30000)
val_images, val_labels = gtsrb_loader.test('grey', 10000)

train_loader = DataLoader(TensorDataset(train_images, train_labels),
                          batch_size=batch_size, shuffle=True)


model = CNN()


for e in range(epochs):
    for (x, y) in tqdm(train_loader):
        y_pred = model.forward_train(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        for param in model.parameters():
            param.data -= eta(e)*param.grad.data
            param.grad.data.zero_()

    train_acc = accuracy(train_images, train_labels)
    val_acc = accuracy(val_images, val_labels)
    print("Epoch {:3} : train_acc = {:5.2f}% ; val_acc = {:5.2f}%"
          .format(e+1, train_acc, val_acc))


def cal_r():
    L = []
    for im in enumerate(val_images):
        y = val_labels[im[0]]
        y_pred = model.forward_eval(im[1].view(1, 1, 40, 40))
        if y_pred.max(1)[1].data[0] != y:
            L.append([im[0], y, y_pred.max(1)[1].data[0]])
    return L


E = cal_r()


def show_r(L):
    for i in range(len(L)):
        show_im(val_images[L[i][0]][0], L[i][1], L[i][2])


def show_im(image, y, y_pred):
    plt.title("prédiction: {} au lieu d'un {}".format(y_pred, y))
    plt.imshow(image.numpy(), cmap='gray')
    plt.show()


def pred_r(n):
    for i in range(n):
        p = random.randrange(50000)
        a = train_images[p]
        b = model.forward_eval(a.view(1, 1, 28, 28))
        plt.title('prédiction: {}'.format(b.max(1)[1].data[0]))
        plt.imshow(train_images[p].numpy(), cmap='gray')
        plt.show()
