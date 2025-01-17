# gtsrb_loader.py
# création et chargement de la base de données

import os
import wget
import shutil
import zipfile
import numpy as np
import pandas as pd
import glob
import torch
from skimage import io, color, transform, exposure
from joblib import Parallel, delayed
import random
import copy


# Téléchargement et décompression des images
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_train_folder():
    train_url = 'http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip'
    if not os.path.exists('../data'):
        os.makedirs('../data')
    if not os.path.exists('../data/Final_Training'):
        print("Downloading the train database...")
        wget.download(train_url, '../data/train.zip')
        print("\nDownload complete.")
        print("Unzipping the train database...")
        zip_ref = zipfile.ZipFile('../data/train.zip', 'r')
        zip_ref.extractall('../data/')
        zip_ref.close()
        print("Unzip complete.")
        shutil.move('../data/GTSRB/Final_Training', '../data/')
        shutil.rmtree('../data/GTSRB')
        os.remove('../data/train.zip')


def get_test_folder():
    test_url = 'http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip'
    test_labels_url = 'http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_GT.zip'
    if not os.path.exists('../data'):
        os.makedirs('../data')
    if not os.path.exists('../data/Final_Test'):
        print("Downloading the test database...")
        wget.download(test_url, '../data/test.zip')
        wget.download(test_labels_url, '../data/test_labels.zip')
        print("\nDownload complete.")
        print("Unzipping the test database...")
        zip_ref = zipfile.ZipFile('../data/test.zip', 'r')
        zip_ref.extractall('../data/')
        zip_ref = zipfile.ZipFile('../data/test_labels.zip', 'r')
        zip_ref.extractall('../data/GTSRB/Final_Test')
        zip_ref.close()
        print("Unzip complete.")
        shutil.move('../data/GTSRB/Final_Test', '../data/')
        shutil.rmtree('../data/GTSRB')
        os.remove('../data/test.zip')
        # Réorganisation du dossier Images/ en sous-dossiers
        data_url = '../data/Final_Test/Images/'
        labels = np.array(pd.read_csv('../data/Final_Test/GT-final_test.csv', sep=';'))[:,7]
        for i in range(0, 43):
            if not os.path.exists(data_url + str(i).zfill(5)):
                os.makedirs(data_url + str(i).zfill(5))
        for i in range(len(labels)):
            image = str(i).zfill(5) + '.ppm'
            label = str(labels[i]).zfill(5)
            os.rename(data_url + image, data_url + label + '/' + image)


# Transformation en tenseur

def traite_image(chemin_image, couleur):
    # Lecture de l'image
    image = io.imread(chemin_image)
    # Redimensionnement en 40x40 pixels
    image = transform.resize(image, (40, 40), mode='wrap')
    # Ajustement local de l'exposition
    if couleur == 'clahe':
        image = exposure.equalize_adapthist(image)
    # Conversion en nuances de gris
    if not couleur == 'rgb':
        image = color.rgb2gray(image)
    return image

def traite_label(chemin_image):
    return int(chemin_image.split('/')[-2])


# Enregistrement des tenseurs

def save_train(images, labels, couleur):
    if not os.path.exists('../data/' + couleur):
        os.makedirs('../data/' + couleur)
    torch.save((images, labels), '../data/' + couleur + '/train.pt')


def save_test(images, labels, couleur):
    if not os.path.exists('../data/' + couleur):
        os.makedirs('../data/' + couleur )
    torch.save((images, labels), '../data/' + couleur + '/test.pt')


# Chargement des tenseurs 

def train(couleur, val_split, num_element=None):  # Couleur : 'rgb', 'grey', 'clahe'
    if not os.path.exists('../data/' + couleur + '/train.pt'):
        chemins_images = glob.glob(os.path.join("..", 'data/Final_Training/Images/', '*/*.ppm'))
        images = Parallel(n_jobs=16)(delayed(traite_image)(path, couleur) for path in chemins_images)
        labels = Parallel(n_jobs=16)(delayed(traite_label)(path) for path in chemins_images)
        mélange(images, labels)
        images = torch.Tensor(images)
        labels = torch.Tensor(labels)
        if couleur == 'rgb':
            images = images.permute(0, 3, 1, 2)
        else:
            images = images.view(len(images), 1, 40, 40)
        save_train(images, labels, couleur)
    images, labels = torch.load('../data/' + couleur + '/train.pt')
    nb_train = round((1 - val_split) * len(images))
    images, labels = images[:nb_train], labels[:nb_train].long()
    return images.to(device), labels.to(device)


def test(couleur, nb_val=12630):
    if not os.path.exists('../data/' + couleur + '/test.pt'):
        chemins_images = glob.glob(os.path.join('../data/Final_Test/Images/', '*/*.ppm'))
        images = Parallel(n_jobs=16)(delayed(traite_image)(path, couleur) for path in chemins_images)
        labels = Parallel(n_jobs=16)(delayed(traite_label)(path) for path in chemins_images)
        images = torch.Tensor(images)
        labels = torch.Tensor(labels).long() 
        if couleur == 'rgb':
            images = images.permute(0, 3, 1, 2)
        else:
            images = images.view(nb_val, 1, 40, 40)
        save_test(images, labels, couleur)
    images, labels = torch.load('../data/' + couleur + '/test.pt')
    images, labels = images[:nb_val], labels[:nb_val].long()
    return images.to(device), labels.to(device)


def val(couleur, val_split, num_element=None):
    images, labels = torch.load('../data/' + couleur + '/train.pt')
    if num_element is not None:
        num_element = len(images)
    nb_train = round((1 - val_split) * len(images))
    images, labels = images[nb_train:num_element], labels[nb_train:num_element].long()
    return images.to(device),  labels.to(device)


def mélange(images, labels):
    rng_state = np.random.get_state()
    np.random.shuffle(images)
    np.random.set_state(rng_state)
    np.random.shuffle(labels)


# Listes des catégories de paneaux symétrisables.
sym_vert = [11, 12, 13, 15, 17, 18, 22, 26, 30, 35]
sym_hori = [1, 5, 12, 15, 17]
sym_diag_droit = [15, 32, 39]
sym_diag_gauch = [15, 32, 38]
sym_mut_vert = [[19, 20], [20, 19], [33, 34], [34, 33], [36, 37], [37, 36], [38, 39], [39, 38]]


# Ajoute tous les symétriques à une BDD et mélange les images.
def ajout_sym():
    for i in range(43):
        print(i)
        if i < 10:
            chemin_images = glob.glob(os.path.join('../data/Final_Training/Images/0000{}/*.ppm'.format(i)))
            chemin_dossier = '../data/Final_Training/Images/0000{}'.format(i)

        else:
            chemin_images = glob.glob(os.path.join('../data/Final_Training/Images/000{}/*.ppm'.format(i)))
            chemin_dossier = '../data/Final_Training/Images/000{}'.format(i)
        n = len(chemin_images)
        if i in sym_vert:
            for j in range(n):
                image = io.imread(chemin_images[j])
                image_sym = image[:, ::-1]
                io.imsave(chemin_dossier + '/vert_{}.ppm'.format(j), image_sym)
        if i in sym_hori:
            for j in range(n):
                image = io.imread(chemin_images[j])
                image_sym = image[::-1, :]
                io.imsave(chemin_dossier + '/hori_{}.ppm'.format(j), image_sym)
        if i in sym_diag_droit:
            for j in range(n):
                image = io.imread(chemin_images[j])
                image_sym = transpose(image[:, ::-1])[:, ::-1]
                io.imsave(chemin_dossier + '/diagd_{}.ppm'.format(j), image_sym)
        if i in sym_diag_gauch:
            for j in range(n):
                image = io.imread(chemin_images[j])
                image_sym = transpose(image)
                io.imsave(chemin_dossier + '/diagg_{}.ppm'.format(j), image_sym)
        if i in [sym_mut_vert[j][0] for j in range(len(sym_mut_vert))]:
            for j in range(n):
                image = io.imread(chemin_images[j])
                image_sym = image[:, ::-1]
                L = [sym_mut_vert[j][0] for j in range(len(sym_mut_vert))]
                dest = L.index(i)
                chemin = '../data/Final_Training/Images/000{}'.format(sym_mut_vert[dest][1])
                io.imsave(chemin + '/mut_{}.ppm'.format(j), image_sym)


def train_sym(couleur):  # Couleur : 'rgb', 'grey', 'clahe'
    if not os.path.exists('../data/' + couleur + '_sym'):
        print('ajout du chemin')
        os.makedirs('../data/' + couleur + '_sym')
    if not os.path.exists('../data/' + couleur + '_sym/train.pt'):
        chemins_images = glob.glob(os.path.join('../data/Final_Training/Images/*/*.ppm'))
        images = Parallel(n_jobs=16)(delayed(traite_image)(path, couleur) for path in chemins_images)
        labels = Parallel(n_jobs=16)(delayed(traite_label)(path) for path in chemins_images)
        mélange(images, labels)
        images = torch.Tensor(images)
        labels = torch.Tensor(labels)
        if couleur == 'rgb':
            images = images.permute(0, 3, 1, 2)
        else:
            images = images.view(len(images), 1, 40, 40)
        torch.save((images, labels), '../data/' + couleur + '_sym/train.pt')
        print('done')


def transpose(tab):
    tab = transform.resize(tab, (40, 40), mode='wrap')
    for i in range(40):
        for j in range(40):
            if j < i:
                tab[j][i], tab[i][j] = copy.copy(tab[i][j]), copy.copy(tab[j][i])
    return tab


# Transforme aléatoirement une image 40X40 par distorsion.
def distorsion(image):
    src = np.array([[0, 0], [0, 40], [40, 40], [40, 0]])
    dst = np.array([[random.randrange(-3, 3), random.randrange(-3, 3)],
                    [random.randrange(-3, 3), 40 - random.randrange(-3, 3)],
                    [40 - random.randrange(-3, 3),40 - random.randrange(-3, 3)],
                    [40 - random.randrange(-3, 3), random.randrange(-3, 3)]])
    disto = transform.ProjectiveTransform()
    disto.estimate(src, dst)
    image_dist = transform.warp(image, disto, output_shape=(40, 40), mode='edge')
    return image_dist


def norm(i,nb):
    if i < 10:
        chemin_images = glob.glob(os.path.join('../data/Final_Training/Images/0000{}/*.ppm'.format(i)))
        chemin_dossier = '../data/Final_Training/Images/0000{}'.format(i)
    else:
        chemin_images = glob.glob(os.path.join('../data/Final_Training/Images/000{}/*.ppm'.format(i)))
        chemin_dossier = '../data/Final_Training/Images/000{}'.format(i)
    n = len(chemin_images)
    if n < nb:
        k = nb // n +1
        for j in range(n):
            image = io.imread(chemin_images[j])
            for m in range(k):
                im = distorsion(image)
                io.imsave(chemin_dossier + '/{}_{}.ppm'.format(j, m), im)


def normalisation(nb):
    for i in range(43):
        print(i)
        norm(i, nb)


def resizee():
    for i in range(43):
        print(i)
        if i < 10:
            chemin_images = glob.glob(os.path.join('../data/Final_Training/Images/0000{}/*.ppm'.format(i)))

        else:
            chemin_images = glob.glob(os.path.join('../data/Final_Training/Images/000{}/*.ppm'.format(i)))
        n = len(chemin_images)
        print(n)
        for j in range(n):
            im = io.imread(chemin_images[j])
            im = transform.resize(im, (40, 40), mode='wrap')
            io.imsave(chemin_images[j], im)


def create():
    chemins_images = []
    for i in range(43):
        if i < 10:
            cont = glob.glob(os.path.join('../data/Final_Training/Images/0000{}/*.ppm'.format(i)))
        else:
            cont = glob.glob(os.path.join('../data/Final_Training/Images/000{}/*.ppm'.format(i)))
        random.shuffle(cont)
        for j in range(2000):
            chemins_images.append(cont[j])
    print(len(chemins_images))
    return chemins_images


def create_dist():
    resizee()
    normalisation(2000)
    A = create()
    print(len(A))


def train_dist(couleur, chemin_images):  # Couleur : 'rgb', 'grey', 'clahe'
    if not os.path.exists('../data/' + couleur + '_dist'):
        os.makedirs('../data/' + couleur + '_dist')
    if not os.path.exists('../data/' + couleur + '_dist/train.pt'):
        images = Parallel(n_jobs=16)(delayed(traite_image)(path, couleur) for path in chemin_images)
        labels = Parallel(n_jobs=1)(delayed(traite_label)(path) for path in chemin_images)
        mélange(images, labels)
        images = torch.Tensor(images)
        labels = torch.Tensor(labels)
        if couleur == 'rgb':
            images = images.permute(0, 3, 1, 2)
        else:
            images = images.view(len(images), 1, 40, 40)
        torch.save((images, labels), '../data/' + couleur + '_dist/train.pt')
        print('done')
