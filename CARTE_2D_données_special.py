# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 14:48:47 2024

@author: arahim
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pathlib import Path
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.colors import Normalize, LogNorm

che= Path(r"C:\Users\arahim\Nextcloud\PhD\simu\test simu coro")
tot_diff_llowfs=np.load(che/'tot_diff_llowfs_diff.npy')
tot_diff_zwfs=np.abs(np.load(che/'tot_diff_zwfs_diff.npy'))
tot_diff_llowfs_zwfs=np.load(che/'tot_diff_llowfs+zwfs_diff.npy')
tot_diff_llowfs_normalized = tot_diff_llowfs*1#np.abs((tot_diff_llowfs - np.mean(tot_diff_llowfs, axis=1, keepdims=True)) / np.std(tot_diff_llowfs, axis=1, keepdims=True))
#tot_diff_zwfs_normalized = np.abs((tot_diff_zwfs - np.mean(tot_diff_zwfs, axis=1, keepdims=True)) / np.std(tot_diff_zwfs, axis=1, keepdims=True))
#tot_diff_llowfs_zwfs_normalized = np.abs((tot_diff_llowfs_zwfs - np.mean(tot_diff_llowfs_zwfs, axis=1, keepdims=True)) / np.std(tot_diff_llowfs_zwfs, axis=1, keepdims=True))

#%%
valeurs_amp=200E-9
zwfs_amp=cube_amp=np.arange(-valeurs_amp, valeurs_amp,((valeurs_amp)/10))
h=np.arange(0,19,1)


x = h # Axe X : 
y = zwfs_amp * 10**9  # Axe Y : 
X, Y = np.meshgrid(x, y) 
#%% ZWFS+LLOWFS
# Calculer les limites globales pour ZWFS

# Utiliser la colormap "rainbow" avec l'échelle du LLOWFS pour ZWFS
#plt.figure('LLOWFS+ZWFS')
global_min_llowfs_zwfs = np.min(tot_diff_llowfs_zwfs)
global_max_llowfs_zwfs = np.max(tot_diff_llowfs_zwfs)

# Créer une figure avec deux sous-graphiques (côte à côte)
# Créer une figure avec deux sous-graphiques (côte à côte)
fig, axs = plt.subplots(1, 3, figsize=(12, 5   ))

# Afficher la première carte LLOWFS+ZWFS avec la colormap "rainbow"
contour1 = axs[0].contourf(X, Y, np.log10(np.abs(tot_diff_llowfs_zwfs.T)), levels=np.linspace(-2,3,100), cmap='magma', vmin=-1,vmax=2)#norm=Normalize(vmin=global_min_llowfs_zwfs, vmax=global_max_llowfs_zwfs))
axs[0].set_title('LLOWFS+ZWFS CIRCULAR')
axs[0].set_xlabel('Spatial frequency (cycl/pup)')
axs[0].set_ylabel('Amplitude of aberration (nm)')
axs[0].set_ylim(0, 180)
axs[0].set_yticks([0, 50, 100, 150])

# Afficher la deuxième carte ZWFS avec la même colormap et les mêmes limites
contour2 = axs[1].contourf(X, Y, np.log10(np.abs(tot_diff_zwfs.T)), levels=np.linspace(-2,3,100), cmap='magma', vmin=-1,vmax=2)# norm=Normalize(vmin=global_min_llowfs_zwfs, vmax=global_max_llowfs_zwfs))
axs[1].set_title('ZWFS CIRCULAR')
axs[1].set_xlabel('Spatial frequency (cycl/pup)')
axs[1].set_ylabel('Amplitude of aberration (nm)')
axs[1].set_ylim(0, 180)
axs[1].set_yticks([0, 50, 100, 150])




contour3 = axs[2].contourf(X, Y, np.log10(np.abs(tot_diff_llowfs.T)), levels=np.linspace(-2,3,100), cmap='magma', vmin=-1,vmax=2)# norm=Normalize(vmin=global_min_llowfs_zwfs, vmax=global_max_llowfs_zwfs))
axs[2].set_title('LLOWFS CIRCULAR')
axs[2].set_xlabel('Spatial frequency (cycl/pup)')
axs[2].set_ylabel('Amplitude of aberration (nm)')
axs[2].set_ylim(0, 180)
axs[2].set_yticks([0, 50, 100, 150])




# Ajouter une colorbar centrée entre les deux sous-graphes
cbar_ax = fig.add_axes([0.915, 0.15, 0.02, 0.7])  # Position centrée entre les sous-graphiques
cbar = fig.colorbar(contour2, cax=cbar_ax, orientation='vertical')
cbar.set_ticks(np.linspace(-2,3,10))
cbar.set_label("log10(Measurement errors nm) ")
#cbar.set_ticks([0, 10, 20, 30, 40]) 
# Ajuster la disposition des éléments pour que tout soit bien aligné
plt.subplots_adjust(wspace=0.4)  # Ajuster l'espace entre les sous-graphiques
plt.show()
#%%
"""
# Utiliser la même colormap "rainbow" mais normalisée aux nouvelles limites ZWFS
contour = plt.contourf(X, Y, np.abs(tot_diff_llowfs_zwfs.T), levels=1000, cmap='viridis', norm=Normalize(vmin=global_min_llowfs_zwfs, vmax=global_max_llowfs_zwfs))

cbar = plt.colorbar(contour)

# Définir les ticks pour la colorbar
#cbar.set_ticks([0, 1, 2])  # Pour afficher seulement 0, 1 et 2
# ou pour l'autre cas :
# cbar.set_ticks([0, 1, 2, 3, 4])  # Pour afficher 0, 1, 2, 3 et 4

# Définir l'étiquette de la colorbar
cbar.set_label("Measurement errors Normalized")


plt.xlabel('Spatial frequency (cycl/pup)')
plt.ylabel('Amplitude of aberration (nm)')
plt.title('Error map normalized LLOWFS+ZWFS SUBARU ')
plt.ylim(0, 180)
plt.yticks([0,50,100,150])
#plt.ylim(0, 200)
#plt.xlim(0, 20) 
#%% ZWFS
# Calculer les limites globales pour ZWFS

# Utiliser la colormap "rainbow" avec l'échelle du LLOWFS pour ZWFS
plt.figure('ZWFS')
global_min_zwfs = np.min(tot_diff_zwfs)
global_max_zwfs = np.max(tot_diff_zwfs)
# Utiliser la même colormap "rainbow" mais normalisée aux nouvelles limites ZWFS
contour = plt.contourf(X, Y, np.abs(tot_diff_zwfs.T), levels=1000, cmap='viridis', norm=Normalize(vmin=global_min_llowfs_zwfs, vmax=global_max_llowfs_zwfs))

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
plt.ylim(0, 180)
plt.yticks([0,50,100,150])
#plt.ylim(0, 200)
#plt.xlim(0, 20)
"""
#%% LLOWFS
"""
# Calculer les limites globales pour LLOWFS
global_min_llowfs = np.min(tot_diff_llowfs_normalized)
global_max_llowfs = np.max(tot_diff_llowfs_normalized)
fig, axs = plt.subplots(1, 1, figsize=(6,5))

contour3 = axs.contourf(X, Y, np.log10(np.abs(tot_diff_llowfs.T)), levels=np.linspace(-2,3,100), cmap='magma', vmin=-1,vmax=2)# norm=Normalize(vmin=global_min_llowfs_zwfs, vmax=global_max_llowfs_zwfs))
axs.set_title('LLOWFS SUBARU')
axs.set_xlabel('Spatial frequency (cycl/pup)')
axs.set_ylabel('Amplitude of aberration (nm)')
axs.set_ylim(0, 180)
axs.set_yticks([0, 50, 100, 150])
#cbar.set_ticks(np.linspace(-2,3,10))
cbar.set_ticks(np.linspace(-2,3,10))
cbar.set_label("log10(Measurement errors nm) ",fontsize=12, labelpad=20)
axs.set_xticks([0,5,10,15])
cbar_ax = fig.add_axes([0.91, 0.15, 0.04, 0.7])
cbar = fig.colorbar(contour3, cax=cbar_ax, orientation='vertical')
"""

"""

# Création de la figure pour LLOWFS
plt.figure('LLOWFS')

# Créer une normalisation basée sur LLOWFS
norm_llowfs = Normalize(vmin=global_min_llowfs, vmax=global_max_llowfs)

# Utiliser contourf pour une carte de chaleur
#plt.contourf(X, Y, (tot_diff_llowfs_normalized.T), levels=50, cmap='rainbow', norm=norm_llowfs)
contour = plt.contourf(X, Y,(np.log10(np.abs(tot_diff_llowfs_normalized.T))), levels=np.linspace(-2,3,100), cmap='magma',  norm=Normalize(vmin=global_min_llowfs_zwfs, vmax=global_max_llowfs_zwfs))

# Créer la colorbar avec des ticks spécifiques
# Créer la colorbar
cbar = plt.colorbar(contour)

# Définir les ticks pour la colorbar

# ou pour l'autre cas :
#cbar.set_ticks([0, 1, 2, 3, 4])  # Pour afficher 0, 1, 2, 3 et 4

# Définir l'étiquette de la colorbar
cbar.set_label("Measurement errors Normalized")

# Ajouter les étiquettes et le titre
plt.xlabel('Spatial frequency (cycl/pup)')
plt.ylabel('Amplitude of aberration (nm)')
plt.title('Error map normalized LLOWFS Circular FQPM')
plt.ylim(0, 180)
plt.yticks([0,50,100,150])
#plt.xlim(0, 370)
"""
