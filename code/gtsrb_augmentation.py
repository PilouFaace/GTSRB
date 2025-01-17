"""Diverses fonctions d'augmentation des BDD de panneaux."""


import numpy as np
import random
from skimage import transform, io
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import os
import glob



def mélange(images, labels):
    rng_state = np.random.get_state()
    np.random.shuffle(images_ext)
    np.random.set_state(rng_state)
    np.random.shuffle(labels_ext)


# Listes des catégories de paneaux symétrisables.
auto_miroir_vertical = [11, 12, 13, 15, 17, 18, 22, 26, 30, 35]
auto_miroir_horizontal = [1, 5, 12, 15, 17]
auto_miroir_double = [12, 15, 17, 32]
mutuel_miroir_vert = [[19,20], [33,34], [36,37], [38,39], [20,19], [34,33], [37,36], [39,38]]


# Ajoute tous les symétriques à une BDD et mélange les images.
def symétries(images, labels):
    labels_num = np.array([np.argmax(i) for i in labels])
    images_ext = np.empty((0, 40, 40), dtype = images.dtype)
    labels_ext = np.empty((0, 43), dtype = labels.dtype)
    for c in range(43):
        print(c)
        images_c = images[labels_num == c]
        images_ext = np.append(images_ext, images_c, axis=0)
        if c in auto_miroir_horizontal:
            images_ext = np.append(images_ext, images_c[:, :, ::-1], axis=0)
        if c in auto_miroir_vertical:
            images_ext = np.append(images_ext, images_c[:, ::-1, :], axis=0)
        if c in auto_miroir_double:
            images_ext = np.append(images_ext, images_c[:, ::-1, ::-1], axis=0)
        nouv_labels = np.full((len(images_ext) - len(labels_ext), 43), np.eye(43)[c])
        labels_ext = np.append(labels_ext, nouv_labels, axis=0)
        if [c, c+1] in mutuel_miroir_vert:
            images_ext = np.append(images_ext, images_c[:, ::-1, :], axis=0)
            nouv_labels = np.full((len(images_ext) - len(labels_ext), 43), np.eye(43)[c+1])
            labels_ext = np.append(labels_ext, nouv_labels, axis=0)
        if [c, c-1] in mutuel_miroir_vert:
            images_ext = np.append(images_ext, images_c[:, ::-1, :], axis=0)
            nouv_labels = np.full((len(images_ext) - len(labels_ext), 43), np.eye(43)[c-1])
            labels_ext = np.append(labels_ext, nouv_labels, axis=0)
    mélange(images_ext, labels_ext)
    return images_ext, labels_ext


# Multiplie par n la taille de la BDD, et mélange les images.
def distorsions(images, labels, n=5):
    labels_dist = labels
    images_dist = images
    for i in range(n-1):
        labels_dist = np.append(labels_dist, labels, axis=0)
    for i in range(n-1):
        images_dist_i = Parallel(n_jobs=4)(delayed(distorsion)(img) for img in images)
        images_dist = np.append(images_dist, np.array(images_dist_i), axis=0)
    mélange(images_dist, labels_dist)
    return images_dist, labels_dist


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
        chemin_images = glob.glob(os.path.join('data/Final_Training_dist/Images/0000{}/*.ppm'.format(i)))
        chemin_dossier = 'data/Final_Training_dist/Images/0000{}'.format(i)
    else:
        chemin_images = glob.glob(os.path.join('data/Final_Training_dist/Images/000{}/*.ppm'.format(i)))
        chemin_dossier = 'data/Final_Training_dist/Images/000{}'.format(i)
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
            chemin_images = glob.glob(os.path.join('data/Final_Training_dist/Images/0000{}/*.ppm'.format(i)))
            chemin_dossier = 'data/Final_Training_dist/Images/0000{}'.format(i)

        else:
            chemin_images = glob.glob(os.path.join('data/Final_Training_dist/Images/000{}/*.ppm'.format(i)))
            chemin_dossier = 'data/Final_Training_dist/Images/000{}'.format(i)
        n = len(chemin_images)
        print(n)
        for j in range(n):
            im = io.imread(chemin_images[j])
            im = transform.resize(im, (40, 40), mode='wrap')
            io.imsave(chemin_images[j], im)



