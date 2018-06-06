"""
Charge la base de données GTSRB


TODO: Enregistrer au format Bytes et diviser par 255 à l'appel (comme MNIST)
"""



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


def mélange(images, labels):
    rng_state = np.random.get_state()
    np.random.shuffle(images)
    np.random.set_state(rng_state)
    np.random.shuffle(labels)

# Téléchargement et décompression des images


def get_train_folder():
    train_url = 'http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip'
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('data/Final_Training'):
        print("Downloading the train database...")
        wget.download(train_url, 'data/train.zip')
        print("\nDownload complete.")
        print("Unzipping the train database...")
        zip_ref = zipfile.ZipFile('data/train.zip', 'r')
        zip_ref.extractall('data/')
        zip_ref.close()
        print("Unzip complete.")
        shutil.move('data/GTSRB/Final_Training', 'data/')
        shutil.rmtree('data/GTSRB')
        os.remove('data/train.zip')

def get_test_folder():
    test_url = 'http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip'
    test_labels_url = 'http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_GT.zip'
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('data/Final_Test'):
        print("Downloading the test database...")
        wget.download(test_url, 'data/test.zip')
        wget.download(test_labels_url, 'data/test_labels.zip')
        print("\nDownload complete.")
        print("Unzipping the test database...")
        zip_ref = zipfile.ZipFile('data/test.zip', 'r')
        zip_ref.extractall('data/')
        zip_ref = zipfile.ZipFile('data/test_labels.zip', 'r')
        zip_ref.extractall('data/GTSRB/Final_Test')
        zip_ref.close()
        print("Unzip complete.")
        shutil.move('data/GTSRB/Final_Test', 'data/')
        shutil.rmtree('data/GTSRB')
        os.remove('data/test.zip')
        # Réorganisation du dossier Images/ en sous-dossiers
        data_url = 'data/Final_Test/Images/'
        labels = np.array(pd.read_csv('data/Final_Test/GT-final_test.csv', sep=';'))[:,7]
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
    if not os.path.exists('data/' + couleur + '_dist'):
        os.makedirs('data/' + couleur + '_dist')
    torch.save((images, labels), 'data/' + couleur + '_dist/train.pt')

def save_test(images, labels, couleur):
    if not os.path.exists('data/' + couleur + '_dist'):
        os.makedirs('data/' + couleur + '_dist')
    torch.save((images, labels), 'data/' + couleur + '_dist/test.pt')


# Chargement des tenseurs 

def train(couleur, val_split, num_element=None):  # Couleur : 'rgb', 'grey', 'clahe'
    if not os.path.exists('../data/' + couleur + '/train.pt'):
        chemins_images = glob.glob(os.path.join("..", 'data/Final_Training/Images/', '*/*.ppm'))
        images = Parallel(n_jobs=4)(delayed(traite_image)(path, couleur) for path in chemins_images)
        labels = Parallel(n_jobs=4)(delayed(traite_label)(path) for path in chemins_images)
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
    return images, labels
print

def test(couleur, nb_val=12630):
    if not os.path.exists('../data/' + couleur + '/test.pt'):
        chemins_images = glob.glob(os.path.join('../data/Final_Test/Images/', '*/*.ppm'))
        images = Parallel(n_jobs=4)(delayed(traite_image)(path, couleur) for path in chemins_images)
        labels = Parallel(n_jobs=4)(delayed(traite_label)(path) for path in chemins_images)
        images = torch.Tensor(images)
        labels = torch.Tensor(labels).long() 
        if couleur == 'rgb':
            images = images.permute(0, 3, 1, 2)
        else:
            images = images.view(nb_val, 1, 40, 40)
        save_test(images, labels, couleur)
    images, labels = torch.load('../data/' + couleur + '/test.pt')
    images, labels = images[:nb_val], labels[:nb_val].long()
    return images, labels


def val(couleur, val_split, num_element=None):
    images, labels = torch.load('../data/' + couleur + '/train.pt')
    if num_element is not None:
        num_element = len(images)
    nb_train = round((1 - val_split) * len(images))
    images, labels = images[nb_train:num_element], labels[nb_train:num_element].long()
    return images, labels


def train_dist(couleur, chemins_images):  # Couleur : 'rgb', 'grey', 'clahe'
    if not os.path.exists('../data/' + couleur + '_dist/train.pt'):
        images = Parallel(n_jobs=4)(delayed(traite_image)(path, couleur) for path in chemins_images)
        labels = Parallel(n_jobs=4)(delayed(traite_label)(path) for path in chemins_images)
        mélange(images, labels)
        images = torch.Tensor(images)
        labels = torch.Tensor(labels)
        if couleur == 'rgb':
            images = images.permute(0, 3, 1, 2)
        else:
            images = images.view(len(images), 1, 40, 40)
        save_train(images, labels, couleur)
        print('done')


def create():
    chemins_images = []
    for i in range(43):
        if i < 10:
            cont = glob.glob(os.path.join('data/Final_Training_dist/Images/0000{}/*.ppm'.format(i)))
        else:
            cont = glob.glob(os.path.join('data/Final_Training_dist/Images/000{}/*.ppm'.format(i)))
        random.shuffle(cont)
        for j in range(2000):
            chemins_images.append(cont[j])
    print(len(chemins_images))
    return chemins_images

