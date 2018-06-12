import argparse
import os
import matplotlib
matplotlib.use('Agg')
import torch
from torch.utils.data import DataLoader, TensorDataset
import gtsrb_loader
import architectures
from tqdm import tqdm
import matplotlib.pyplot as plt
import plot

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Parameters parsing
parser = argparse.ArgumentParser()
parser.add_argument("model", type=str)
parser.add_argument("color", type=str)
parser.add_argument("-sym", action="store_true")
parser.add_argument("-dist", action="store_true")
parser.add_argument("-num", type=int)
parser.add_argument("-split", type=str, default="1/6")
parser.add_argument("-lr", type=float)
parser.add_argument("-e", type=int)
parser.add_argument("-bs", type=int)
parser.add_argument("-s", "--save", action="store_true")
args = parser.parse_args()

model_name = args.model
if args.sym:
    dset_name = args.color + "_sym"
elif args.dist:
    dset_name = args.color + "_dist"
else:
    dset_name = args.color
num_img = args.num
val_split = eval(args.split)
save_model = args.save

# Model instanciation
model = getattr(architectures, model_name)()
model = model.to(device)

# Loads model hyperparameters (if not specified in args)
batch_size = args.bs if args.bs else model.batch_size
lr = args.lr if args.lr else model.lr
epochs = args.e if args.e else model.epochs


# Loads model functions
loss_fn = model.loss_fn
optimizer = model.optimizer


# Loads the training database, and splits it in into `train` and `val`.
train_images, train_labels = gtsrb_loader.train(dset_name, val_split, num_img)
val_images, val_labels = gtsrb_loader.val(dset_name, val_split, num_img)
test_images, test_labels = gtsrb_loader.test(args.color)
num_train = len(train_images)
num_val = len(val_images)
print(num_val)


# DataLoader of the `train` images
train_loader = DataLoader(TensorDataset(train_images, train_labels),
                          batch_size=batch_size,
                          shuffle=True)

num_batches = len(train_loader)


# (computing the accuracy mini-batch after mini-batch avoids memory overload)
def accuracy(images, labels):
    count = 0
    train_loader2 = DataLoader(TensorDataset(images, labels), batch_size=256)
    for (x, y) in tqdm(train_loader2):
        y_pred = model.eval()(x)
        count += (y_pred.max(1)[1] == y).double().sum()
    return 100 * count / len(images)


# Computes the loss of the model.
# (computing the loss mini-batch after mini-batch avoids memory overload)
def big_loss(images, labels):
    data = TensorDataset(images, labels)
    loader = DataLoader(data, batch_size=100, shuffle=False)
    count = 0
    for (x, y) in loader:
        y_pred = model.eval()(x)
        count += len(x) * loss_fn(y_pred, y).item()
    return count / len(images)


def save_data(liste, name):
    with open('../models/' + dset_name + '/' + model_name + name + str(lr), 'w') as filehandle:
        for listitem in liste:
            filehandle.write('%s\n' % listitem)

# NETWORK TRAINING
# ----------------
# Custom progress bar.
'''def bar(data, e):
    epoch = f"Epoch {e+1}/{epochs}"
    left = "{desc}: {percentage:3.0f}%"
    right = "{elapsed} - ETA:{remaining} - {rate_fmt}"
    bar_format = left + " |{bar}| " + right
    return tqdm(data, desc=epoch, ncols=74, unit='b', bar_format=bar_format)'''


train_accs, val_accs = [], []
train_losses, val_losses = [], []
test_accs = []

try:
    # Main loop over each epoch
    for e in range(epochs):

        # Secondary loop over each mini-batch
        for (x, y) in tqdm(train_loader):

            # Computes the network output
            y_pred = model.train()(x)
            loss = loss_fn(y_pred, y)

            # Optimizer step
            model.zero_grad()
            loss.backward()
            optimizer.step()

        # Calculates accuracy and loss on the train database.
        train_acc = accuracy(train_images, train_labels).item()
        train_loss = big_loss(train_images, train_labels)
        train_accs.append(train_acc)
        train_losses.append(train_loss)

        # Calculates accuracy and loss on the validation database.
        val_acc = accuracy(val_images, val_labels).item()
        val_loss = big_loss(val_images, val_labels)
        val_accs.append(val_acc)
        val_losses.append(val_loss)
        test_acc = accuracy(test_images, test_labels).item()
        test_accs.append(test_acc)
        # Prints the losses and accs at the end of each epoch.
        print("Epoch {:3} : train_acc = {:5.2f}% ; val_acc = {:5.2f}%"
              .format(e+1, train_acc, val_acc))
    print(test_acc)

# Allows to manually interrupt the training (early stopping).
except KeyboardInterrupt:
    pass

# Saves the network if stated.
if save_model:
    if not os.path.exists("../models/" + dset_name):
        os.makedirs("../models/" + dset_name)
    path = os.path.join("../models/" + dset_name + '/' + model_name + str(lr) + "_" + str(epochs) + ".pt")
    if model_name != "VGG" and model_name != "VGG_bn" :
        torch.save(model, path)
    #Saves the accs history graph
    path = os.path.join("../models/" + dset_name + '/')
    plot.train_history(train_accs, val_accs)
    plt.savefig(path + model_name + "_" + str(lr) + "_" + str(epochs) + ".png", transparent=True)
    # save_data(train_accs, 'train_accs')
    # save_data(val_accs, 'cal_accs')
    # save_data(test_accs, 'test_accs')


# def discriminator_performance(x, y):
#     y_pred = model.eval()(x)
#     faux_pos = ((y_pred.max(1)[1] != y) * (y_pred.max(1)[1] == 0))
#     faux_pos = faux_pos.double().sum()
#     faux_neg = ((y_pred.max(1)[1] != y) * (y_pred.max(1)[1] == 1))
#     faux_neg = faux_neg.double().sum()
#     total = (y_pred.max(1)[1] != y).double().sum()
#     return (faux_pos, faux_neg, total)
path = os.path.join("../models/" + dset_name + '/')