import glob
import os
from skimage import io, color, transform, exposure
import numpy as np
import copy
import matplotlib.pyplot as plt
import torch
from joblib import Parallel, delayed

sym_vert = [11, 12, 13, 15, 17, 18, 22, 26, 30, 35]
sym_hori = [1, 5, 12, 15, 17]
sym_diag_droit = [15, 32, 39]
sym_diag_gauch = [15, 32, 38]
sym_mut_vert = [[19, 20], [20, 19], [33, 34], [34, 33], [36, 37], [37, 36], [38, 39], [39, 38]]
chemin = [glob.glob(os.path.join('data/Final_Training/Images/00001/*.ppm'))[50]]


def transpose(tab):
    tab = transform.resize(tab, (40, 40), mode='wrap')
    for i in range(40):
        for j in range(40):
            if j < i:
                tab[j][i], tab[i][j] = copy.copy(tab[i][j]), copy.copy(tab[j][i])
    return tab


def prep_image(chemin_image, couleur):
    image = io.imread(chemin_image)
    image = transform.resize(image, (40, 40), mode='wrap')
    if couleur == 'grey':
        image = color.rgb2gray(image)
    else:
        image = color.rgb2gray(image)
        image = exposure.equalize_adapthist(image)
    return image


def prep_label(chemin_image):
    return int(chemin_image.split('/')[-2])


def ajout_sym():
    for i in range(43):
        print(i)
        if i < 10:
            chemin_images = glob.glob(os.path.join('data/Final_Training_sym/Images/0000{}/*.ppm'.format(i)))
            chemin_dossier = 'data/Final_Training_sym/Images/0000{}'.format(i)

        else:
            chemin_images = glob.glob(os.path.join('data/Final_Training_sym/Images/000{}/*.ppm'.format(i)))
            chemin_dossier = 'data/Final_Training_sym/Images/000{}'.format(i)
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
                chemin = 'data/Final_Training_sym/Images/000{}'.format(sym_mut_vert[dest][1])
                io.imsave(chemin + '/mut_{}.ppm'.format(j), image_sym)


def compteBDD():
    for i in range(10):
        chemin_images = glob.glob(os.path.join('data/Final_Training_dist/Images/0000{}/*.ppm'.format(i)))
        print((i, len(chemin_images)))
    for i in range(10, 43):
        chemin_images = glob.glob(os.path.join('data/Final_Training_dist/Images/000{}/*.ppm'.format(i)))
        print((i, len(chemin_images)))


def BDD(couleur):
    if not os.path.exists('data/' + couleur):
        os.makedirs('data/' + couleur)
    chemin_images = glob.glob(os.path.join('data/Final_Training/Images/*/*.ppm'))
    L = []
    M = []
    n = len(chemin_images)
    #L.append(torch.Tensor(prep_image(chemin_images[i], couleur)))
    L = Parallel(n_jobs=4)(delayed(prep_image)(chemin_images[i], couleur) for i in range(n))
    M = Parallel(n_jobs=4)(delayed(prep_label)(chemin_images[i]) for i in range(n))
    #M.append(torch.Tensor([prep_label(chemin_images[i])]))

    torch.save((torch.Tensor(L), torch.Tensor(M)), 'data/' + couleur + '/train.pt')


def BDDT(couleur):
    if not os.path.exists('data/' + couleur):
        os.makedirs('data/' + couleur)
    chemin_images = glob.glob(os.path.join('data/Final_Test/Images/*/*.ppm'))
    L = []
    M = []
    n = len(chemin_images)
    L = Parallel(n_jobs=4)(delayed(prep_image)(chemin_images[i], couleur) for i in range(n))
    M = Parallel(n_jobs=4)(delayed(prep_label)(chemin_images[i]) for i in range(n))
    torch.save((torch.Tensor(L), torch.Tensor(M)), 'data/' + couleur + '/test.pt')


