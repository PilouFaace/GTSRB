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

# Parsing des paramètres
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

# Création du modèle
model = getattr(architectures, model_name)()
model = model.to(device)

# Charge les hyperparamètres du modèle
batch_size = args.bs if args.bs else model.batch_size
lr = args.lr if args.lr else model.lr
epochs = args.e if args.e else model.epochs


# Charge les fonctions du modèle
loss_fn = model.loss_fn
optimizer = model.optimizer


# Charge les images et les divise en 'train' et 'val'
train_images, train_labels = gtsrb_loader.train(dset_name, val_split, num_img)
val_images, val_labels = gtsrb_loader.val(dset_name, val_split, num_img)
test_images, test_labels = gtsrb_loader.test(args.color)
num_train = len(train_images)
num_val = len(val_images)
print(num_val)


# DataLoader des images `train`
train_loader = DataLoader(TensorDataset(train_images, train_labels),
                          batch_size=batch_size,
                          shuffle=True)

num_batches = len(train_loader)


# (calcul la justesse par mini-batch pour éviter une saturation de la mémoire)
def accuracy(images, labels):
    count = 0
    train_loader2 = DataLoader(TensorDataset(images, labels), batch_size=256)
    for (x, y) in tqdm(train_loader2):
        y_pred = model.eval()(x)
        count += (y_pred.max(1)[1] == y).double().sum()
    return 100 * count / len(images)


# Calcul les erreurs du modèle
def big_loss(images, labels):
    data = TensorDataset(images, labels)
    loader = DataLoader(data, batch_size=100, shuffle=False)
    count = 0
    for (x, y) in loader:
        y_pred = model.eval()(x)
        count += len(x) * loss_fn(y_pred, y).item()
    return count / len(images)


# ENTRAINEMENT DU RÉSEAU


train_accs, val_accs = [], []
train_losses, val_losses = [], []
test_accs = []

try:
    # Boucle principale
    for e in range(epochs):

        # Boucle secondaire (mini_batch)
        for (x, y) in tqdm(train_loader):

            # Calcule la sortie du réseau
            y_pred = model.train()(x)
            loss = loss_fn(y_pred, y)

            # Optimiseur
            model.zero_grad()
            loss.backward()
            optimizer.step()

        # Calcule la justesse et les erreurs de la base de données train
        train_acc = accuracy(train_images, train_labels).item()
        train_loss = big_loss(train_images, train_labels)
        train_accs.append(train_acc)
        train_losses.append(train_loss)

        # Calcule la justesse et les erreurs de la base de données val
        val_acc = accuracy(val_images, val_labels).item()
        val_loss = big_loss(val_images, val_labels)
        val_accs.append(val_acc)
        val_losses.append(val_loss)

        # Affiche les valeurs après chaque epoch.
        print("Epoch {:3} : train_acc = {:5.2f}% ; val_acc = {:5.2f}%"
              .format(e+1, train_acc, val_acc))
    print(test_acc)

# Permet un arrêt manuel (early stopping)
except KeyboardInterrupt:
    pass

# Sauvegarde le modèle si établit
if save_model:
    if not os.path.exists("../models/" + dset_name):
        os.makedirs("../models/" + dset_name)
    path = os.path.join("../models/" + dset_name + '/' + model_name + str(lr) + "_" + str(epochs) + ".pt")
    if model_name != "VGG" and model_name != "VGG_bn":
        torch.save(model, path)
    # Sauvegarde la courbe
    path = os.path.join("../models/" + dset_name + '/')
    plot.train_history(train_accs, val_accs)
    plt.savefig(path + model_name + "_" + str(lr) + "_" + str(epochs) + ".png", transparent=True)
    plt.clf()
    plot.train_history1(train_losses, val_losses)
    plt.savefig(path + model_name + "_" + "losses" + str(lr) + "_" + str(epochs) + ".png", transparent=True)
