# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 12:19:23 2023

@author: moi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1
from sklearn import preprocessing

wavelength = 0.7  # Longueur d'onde de la lumière en micromètres

# Taille de la grille
grid_size = 1024

# Rayon de l'ouverture circulaire
circle_radius = 40
x0=grid_size/2
y0=grid_size/2

# Création d'une grille de points dans le plan XY
x1 = np.linspace(0, grid_size-1, grid_size)
y1 = np.linspace(0, grid_size-1, grid_size)
x, y = np.meshgrid(x1, y1)

#Création de l'ouverture circulaire 
aperture_radius = (x-x0)**2 + (y-y0)**2 <= circle_radius**2
theta = np.sqrt(x**2 + y**2)
theta_range = np.linspace(-np.pi, np.pi,1024)

P=np.zeros((grid_size,grid_size))
P[aperture_radius]=1
plt.figure('plan A')
plt.imshow(P)
plt.colorbar(label='Intensité')

E_A=1*P

E_B=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(E_A)))
I_B=np.abs(E_B)**2

plt.figure('plan B')
plt.imshow(np.log10(I_B))
plt.colorbar(label='Intensité')

E_C=np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(E_B)))
I_C=np.abs(E_C)**2
plt.figure('plan C avant lyot stop')
plt.imshow(I_C**0.25)
plt.colorbar(label='Intensité')

#creation du LS
radius2 = (x-x0)**2 + (y-y0)**2 <= (circle_radius)**2
P_C=np.zeros((grid_size,grid_size))
P_C[radius2]=1
plt.figure('LS')
plt.imshow(P_C)
plt.colorbar(label='Intensité')

#Lyot stop
E_C2=E_C*(P_C)
I_C2=np.abs(E_C2)**2
plt.figure('plan C apres lyot stop')
plt.imshow(I_C2**0.25)
plt.colorbar(label='Intensité')

#PLAN D 
E_D=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(E_C2),norm='ortho'))
I_D=np.abs(E_D)**2
plt.figure('PLAN D')
plt.imshow(I_D**0.25)

plt.colorbar(label='Intensité')




#Normalisation Intensité avec Intensité max

liste_2D = I_D
# Trouver le maximum dans la liste 2D
diviseur =max(max(row) for row in liste_2D)
I_Dnorm=liste_2D/diviseur
plt.figure('PLAN D normalisé avec le max')
plt.imshow(I_Dnorm)
plt.colorbar(label='Intensité')  

liste_2 = I_B
# Trouver le maximum dans la liste 2D
diviseur2 =max(max(row) for row in liste_2)
I_Bnorm=liste_2/diviseur2
plt.figure('PLAN B normalisé avec le max')
plt.imshow(I_Bnorm)
plt.colorbar(label='Intensité')

# Diviser chaque élément de la liste 2D par le diviseur
# Afficher l'image et les coupes radiales
plt.figure(figsize=(12, 6))
plt.plot(np.log10(I_Bnorm[grid_size//2,grid_size//2:grid_size]), color='blue', alpha=10,label='plan D')

plt.legend()
plt.title('Coupes radiales')
plt.xlabel('Distance depuis le centre')
plt.ylabel('Intensité')
plt.show()

#facteur d'echelle entre plan B(avant coro) et plan D(apres coro)
Facteur= np.around(I_Bnorm /I_Dnorm)
plt.figure('Facteur d echelle')
plt.plot((Facteur[grid_size//2,grid_size//2:grid_size]))

plt.show()

