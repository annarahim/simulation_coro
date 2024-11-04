# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 13:57:42 2024

@author: arahim
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pathlib import Path
from matplotlib import cm
from matplotlib.colors import Normalize

che= Path(r"C:\Users\arahim\Nextcloud\PhD\simu\test simu coro")
tot_diff_llowfs=np.load(che/'tot_diff_llowfs.npy')
tot_diff_zwfs=np.load(che/'tot_diff_zwfs.npy')
tot_diff_llowfs_zwfs=np.load(che/'tot_diff_llowfs_zwfs.npy')



data_min = np.min(tot_diff_llowfs)
data_max = np.max(tot_diff_llowfs)

# Normaliser les données entre 0 et 1
#normalized_data = (tot_diff_llowfs - data_min) / (data_max - data_min)



valeurs_amp=200E-9
zwfs_amp=cube_amp=np.arange(-valeurs_amp, valeurs_amp,((valeurs_amp)/100))
h=np.arange(0,380,10)
valeurs_a_exclure = {50, 100, 150, 200,230, 250, 300, 350}

x = h 
#x = np.array([val for val in h if val not in valeurs_a_exclure])
y = zwfs_amp * 10**9  
X, Y = np.meshgrid(x, y)  
#%% ZWFS
# Calculer les limites globales pour ZWFS

# Utiliser la colormap "rainbow" avec l'échelle du LLOWFS pour ZWFS
plt.figure('ZWFS')
global_min_zwfs = np.min(tot_diff_zwfs)
global_max_zwfs = np.max(tot_diff_zwfs)
# Utiliser la même colormap "rainbow" mais normalisée aux nouvelles limites ZWFS
contour = plt.contourf(X, Y, (np.abs(tot_diff_zwfs.T)), levels=100, cmap='inferno',norm=Normalize(vmin=global_min_zwfs, vmax=global_max_zwfs))

cbar = plt.colorbar(contour)

# Définir les ticks pour la colorbar
#cbar.set_ticks([0, 1, 2])  # Pour afficher seulement 0, 1 et 2
# ou pour l'autre cas :
# cbar.set_ticks([0, 1, 2, 3, 4])  # Pour afficher 0, 1, 2, 3 et 4

# Définir l'étiquette de la colorbar
cbar.set_label("Measurement errors Normalized")


plt.xlabel('Spatial frequency (cycl/pup)')
plt.ylabel('Amplitude of aberration (nm)')
plt.title('Error map normalized ZWFS SUBARU ')
#plt.ylim(0, 180)
#plt.yticks([0,50,100,150])
#plt.ylim(0, 200)
#plt.xlim(0, 20)
#%% LLOWFS
# Calculer les limites globales pour LLOWFS
global_min_llowfs = np.min(tot_diff_llowfs)
global_max_llowfs = np.max(tot_diff_llowfs)

# Création de la figure pour LLOWFS
plt.figure('LLOWFS')

# Créer une normalisation basée LLOWFS
norm_llowfs = Normalize(vmin=global_min_llowfs, vmax=global_max_llowfs)

#plt.contourf(X, Y, (tot_diff_llowfs_normalized.T), levels=50, cmap='rainbow', norm=norm_llowfs)
contour = plt.contourf(X, Y,np.log10(np.abs(tot_diff_llowfs.T)), levels=100, cmap='inferno', vmin=0, vmax=3, norm=Normalize(vmin=global_min_zwfs, vmax=global_max_zwfs))

# Créer la colorbar avec des ticks spécifiques
# Créer la colorbar
cbar = plt.colorbar(contour)#,vmin=0,vmax=6)

# Définir les ticks pour la colorbar

# ou pour l'autre cas :


cbar.set_label("Measurement errors Normalized")


plt.xlabel('Spatial frequency (cycl/pup)')
plt.ylabel('Amplitude of aberration (nm)')
plt.title('Error map normalized LLOWFS SUBARU FQPM')
#plt.ylim(0, 200)
#plt.xlim(0, 370)


#%%

"""
tot_diff_llowfs_normalized = np.abs((tot_diff_llowfs - np.mean(tot_diff_llowfs, axis=1, keepdims=True)) / np.std(tot_diff_llowfs, axis=1, keepdims=True))
tot_diff_zwfs_normalized = np.abs((tot_diff_zwfs - np.mean(tot_diff_zwfs, axis=1, keepdims=True)) / np.std(tot_diff_zwfs, axis=1, keepdims=True))
tot_diff_llowfs_zwfs_normalized = np.abs((tot_diff_llowfs_zwfs - np.mean(tot_diff_llowfs_zwfs, axis=1, keepdims=True)) / np.std(tot_diff_llowfs_zwfs, axis=1, keepdims=True))

#%%
valeurs_amp=200E-9
zwfs_amp=cube_amp=np.arange(-valeurs_amp, valeurs_amp,((valeurs_amp)/100))
h=np.arange(0,380,10)



x = h # Axe X : 
y = zwfs_amp * 10**9  # Axe Y : 
# Grille pour les axes
X, Y = np.meshgrid(x, y)  # 
#%% LLOWFS
# Calculer les limites globales pour LLOWFS
global_min_llowfs = np.min(tot_diff_llowfs_normalized)
global_max_llowfs = np.max(tot_diff_llowfs_normalized)

# Création de la figure pour LLOWFS
plt.figure('LLOWFS')

# Créer une normalisation basée sur LLOWFS
norm_llowfs = Normalize(vmin=global_min_llowfs, vmax=global_max_llowfs)

# Utiliser contourf pour une carte de chaleur
#plt.contourf(X, Y, (tot_diff_llowfs_normalized.T), levels=50, cmap='rainbow', norm=norm_llowfs)
contour = plt.contourf(X, Y, (tot_diff_llowfs_normalized.T), levels=100, cmap='rainbow', norm=norm_llowfs)

# Créer la colorbar avec des ticks spécifiques
# Créer la colorbar
cbar = plt.colorbar(contour)

# Définir les ticks pour la colorbar

# ou pour l'autre cas :
cbar.set_ticks([0, 1, 2, 3, 4])  # Pour afficher 0, 1, 2, 3 et 4


cbar.set_label("Measurement errors Normalized")

plt.xlabel('Spatial frequency (cycl/pup)')
plt.ylabel('Amplitude of aberration (nm)')
plt.title('Error map normalized LLOWFS SUBARU FQPM')
plt.ylim(0, 200)
plt.xlim(0, 370)

#%% ZWFS
# Calculer les limites globales pour ZWFS
global_min_zwfs = np.min(tot_diff_zwfs_normalized)
global_max_zwfs = np.max(tot_diff_zwfs_normalized)

plt.figure('ZWFS')

contour = plt.contourf(X, Y, (tot_diff_zwfs_normalized.T), levels=100, cmap='rainbow', norm=Normalize(vmin=0, vmax=global_max_llowfs))

# Créer la colorbar avec des ticks spécifiques
# Créer la colorbar
cbar = plt.colorbar(contour)

# Définir les ticks pour la colorbar
cbar.set_ticks([0, 1, 2])  # Pour afficher seulement 0, 1 et 2
# ou pour l'autre cas :
# cbar.set_ticks([0, 1, 2, 3, 4])  # Pour afficher 0, 1, 2, 3 et 4

cbar.set_label("Measurement errors Normalized")

plt.xlabel('Spatial frequency (cycl/pup)')
plt.ylabel('Amplitude of aberration (nm)')
plt.title('Error map normalized ZWFS SUBARU ')
plt.ylim(0, 200)
plt.xlim(0, 370)

#%% LLOWFS + ZWFS
# Calculer les limites globales pour LLOWFS+ZWFS
global_min_llowfs_zwfs = np.min(tot_diff_llowfs_zwfs_normalized)
global_max_llowfs_zwfs = np.max(tot_diff_llowfs_zwfs_normalized)

plt.figure('LLOWFS+ZWFS')


# Appliquer la colormap "rainbow" avec les mêmes nuances issues du LLOWFS
contour = plt.contourf(X, Y, (tot_diff_llowfs_zwfs_normalized.T), levels=100, cmap='rainbow', norm=Normalize(vmin=0, vmax=global_max_llowfs))

# Créer la colorbar avec des ticks spécifiques
# Créer la colorbar
cbar = plt.colorbar(contour)

# Définir les ticks pour la colorbar
cbar.set_ticks([0, 1, 2])  # Pour afficher seulement 0, 1 et 2
# ou pour l'autre cas :
# cbar.set_ticks([0, 1, 2, 3, 4])  # Pour afficher 0, 1, 2, 3 et 4

cbar.set_label("Measurement errors Normalized")

plt.xlabel('Spatial frequency (cycl/pup)')
plt.ylabel('Amplitude of aberration (nm)')
plt.title('Error map normalized LLOWFS+ZWFS SUBARU FQPM')
plt.ylim(0, 200)
plt.xlim(0, 370)

plt.show()
"""
"""
#%% LLLOWFS
# Calculer les limites globales
global_min_llowfs = np.min(tot_diff_llowfs_normalized)
global_max_llowfs = np.max(tot_diff_llowfs_normalized)
plt.figure('LLOWFS')
plt.contourf(X, Y, (tot_diff_llowfs_normalized.T), levels=50, cmap='rainbow', vmin=global_min_llowfs, vmax=global_max_llowfs)  # Utiliser vmin et vmax
plt.colorbar(label="Différences Z Normalisées")

plt.xlabel('Fréquence spatiale')
plt.ylabel('Amplitude d’aberration')
plt.title('Carte d’erreur LLOWFS Circular Normalisée')
plt.ylim(0, 200)
plt.xlim(0,370)
#%%ZWFS

# Calculer les limites globales
global_min_zwfs = np.min(tot_diff_zwfs_normalized)
global_max_zwfs = np.max(tot_diff_zwfs_normalized)
plt.figure('ZWFS')
plt.contourf(X, Y, (tot_diff_zwfs_normalized.T), levels=50, cmap='rainbow', vmin=global_min_zwfs, vmax=global_max_zwfs)  # Utiliser vmin et vmax
plt.colorbar(label="Différences Z Normalisées")

plt.xlabel('Fréquence spatiale')
plt.ylabel('Amplitude d’aberration')
plt.title('Carte d’erreur ZWFS Circular Normalisée')
plt.ylim(0, 200)
plt.xlim(0,370)
#%% LLOWFS+ZWFS
# Calculer les limites globales
global_min_llowfs_zwfs = np.min(tot_diff_llowfs_zwfs_normalized)
global_max_llowfs_zwfs = np.max(tot_diff_llowfs_zwfs_normalized)
plt.figure('LLOWFS+ZWFS')
plt.contourf(X, Y, (tot_diff_llowfs_zwfs_normalized.T), levels=50, cmap='rainbow', vmin=global_min_llowfs_zwfs, vmax=global_max_llowfs_zwfs)  # Utiliser vmin et vmax
plt.colorbar(label="Différences Z Normalisées")

plt.xlabel('Fréquence spatiale')
plt.ylabel('Amplitude d’aberration')
plt.title('Carte d’erreur LLOWFS+ZWFS Circular Normalisée')
plt.ylim(0, 200)
plt.xlim(0,370)
"""