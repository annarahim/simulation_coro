# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 11:43:23 2023

@author: moi
"""

from sklearn import preprocessing
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

from matplotlib.patches import Circle

wavelength=0.7
grid_size=1024
circle_radius = 5
corners=[(0,grid_size),(grid_size,grid_size),(grid_size,0),(0,0)]

x0=grid_size/2
y0=grid_size/2
E0=1
# Création d'une grille de points dans le plan XY
x = np.linspace(0, 1024, grid_size)
y = np.linspace(0, 1024, grid_size)
x, y = np.meshgrid(x, y)

C=((x-x0)**2 + (y-y0)**2 ) <= circle_radius**2
r = np.sqrt(x**2 + y**2)

P0 = ((x-1024)**2 + (y-1024)**2 ) <= circle_radius**2
P1=((x-0)**2 + (y-0)**2) <= circle_radius**2
P2=(x-1024)**2 + (y-0)**2<= circle_radius**2
P3= (x-0)**2 + (y-1024)**2<= circle_radius**2
P=P0+P1+P2+P3
    
def electric_field_circular_aperture(r, P, wavelength):
    
    k = 2 * np.pi / wavelength
    return E0*P*np.exp(1j*k*r)
electric_A = electric_field_circular_aperture(r, P, wavelength)

def intensity_distribution(r, P, wavelengt):
    
    E = electric_field_circular_aperture(r, P, wavelength)
    return np.abs(E)**2
intensite_A = intensity_distribution(r, P, wavelength)


plt.figure()
plt.imshow(intensite_A)
plt.title('Plan A')
plt.xlabel('Position X')
plt.ylabel('Position Y')
plt.colorbar(label='Intensité')
plt.show()

#champ electrique plan B
N =10 # Nombre de points d'échantillonnage
L = 100.0  # Longueur totale de l'intervalle d'échantillonnage
dx = L / N  # Espacement entre les points d'échantillonnage
freq = np.fft.fftfreq(N, dx)


    
electric_B1 = (np.fft.fft2(intensite_A))


intensite_B1 = np.fft.fftshift(np.abs(electric_B1)**2)


"""
def tache_airy(r, electric_B):
    intensite = np.abs(electric_B)**2
    return intensite

r = np.sqrt(x0**2 + y0**2)
intensite = tache_airy(r, electric_B)

I_normalisees = preprocessing.normalize(intensite_B1)
"""
plt.figure()
plt.imshow((intensite_B1))
plt.colorbar()
plt.title('Plan B')
plt.xlabel('Position X')
plt.ylabel('Position Y')
plt.colorbar(label='Intensité')
plt.show()