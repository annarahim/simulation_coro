# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:29:20 2024

@author: arahim
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pathlib import Path
from matplotlib import cm

cube_amp=np.arange(-200E-9, 200E-9,((200E-9)*2/100))
valeurs_amp=200E-9
zwfs_amp=cube_amp=np.arange(-valeurs_amp, valeurs_amp,((valeurs_amp)/10)) #pour llowfs pur le reste mettre ca np.arange(-200E-9, 200E-9,((200E-9)*2/100)) : #np.arange(-valeurs_amp, valeurs_amp,((valeurs_amp)/100))
#ZWFS\CIRCULAR\données _circular_sinx

#COMPARAISON cycle_pup\LLOWFS\CIRCULAR\FQPM\N_3
#COMPARAISON cycle_pup\ZWFS+LLOWFS\CIRCULAR\FQPM\données C FQPM sin x
chemin = Path(r"C:\Users\arahim\Nextcloud\PhD\simu\COMPARAISON cycle_pup\LLOWFS\CIRCULAR\VORTEX\c=20")
#nombre eleement dans le dossier (-1)=?"comande à rajouter"
#dernier_element=chemin.name
nombre_fichiers = len([f for f in chemin.glob('*') if f.is_file()])  # Compte  les fichiers

path=chemin

# Récupérer les 4 derniers noms de dossiers
last_five_folders = list(path.parts)[-4:]

# Afficher les résultats
print(last_five_folders)
#a1=np.load(chemin / f'test_zwfs0.txt.npy') 
#plt.plot(zwfs_amp,a1)
#%%
h=0
x_lin_deb=[]
x_lin_fin=[]
sigma_test=[]
taille_bloc = len(cube_amp)
ecarts = []
y_tot=[]
# Tableaux pour stocker les écarts à chaque position cible
ecart_min = []
ecart_midd = []
ecart_max = []
# Définir la plage à sélectionner
x_min = -10E-9
x_max = 10E-9

# Créer un masque pour sélectionner les éléments dans la plage
mask = (zwfs_amp >= x_min) & (zwfs_amp <= x_max)
tot_diff=[]
freq_spatial=[]

#%%


seuil = 0.5 # À définir selon ton seuil

limite_positive_x=[]
limite_negative_x=[]
x_target_values = [200]  # Par exemple
#%%
differences_specifiques=[]

# Récupérer les différences uniquement pour les indices spécifiés

#%%
plt.figure()
h=1
valeurs_a_exclure = {50, 100, 150, 200,230, 250, 300, 350}
for i in range(0, (nombre_fichiers)-1, 1):
   
    a1=np.load(chemin / f'test_zwfs{h}.txt.npy') 
     
    x = zwfs_amp*10**9#[mask]
    y = a1*10**9#[mask ]
    differences = [a - b for a, b in zip(y, x)]
    tot_diff.append(differences) 
    y_tot.append(y)
    plt.plot(x,differences,label=f'{h}')
    #plt.plot(x, differences, label=f'{h}')
    #if h not in valeurs_a_exclure:
       
        
     
    
    #plt.plot(x,differences,label=f'{h }')
    #plt.plot(x,y,label=f'{h }')
    #plt.legend()
    #plt.legend(fontsize=8) 
    h+=1
    
tot_diff = np.array((tot_diff) ) # tot_diff devient un tableau 2D
plt.legend()
np.save('tot_diff_llowfs_diff',tot_diff)
#%%

#%%# Normalisation des données dans tot_diff cad (data-mean(data))/(ecart_type) 
"""#AFFICHAGE 
tot_diff_normalized = (tot_diff - np.mean(tot_diff, axis=1, keepdims=True)) / np.std(tot_diff, axis=1, keepdims=True)

# Calculer les limites globales
global_min = np.min(tot_diff_normalized)
global_max = np.max(tot_diff_normalized)

# Création de la carte de chaleur avec les données normalisées
plt.figure(figsize=(8, 6))
x = np.linspace(0,h,(nombre_fichiers)-1) # Axe X : 38 valeurs
y = zwfs_amp * 10**9  # Axe Y : 201 valeurs
# Grille pour les axes
X, Y = np.meshgrid(x, y)  # Créer une grille pour correspondre aux dimensions (38, 201)

# Utiliser contourf pour une carte de chaleur
plt.contourf(X, Y, (tot_diff_normalized.T), levels=50, cmap='rainbow', vmin=global_min, vmax=global_max)  # Utiliser vmin et vmax
plt.colorbar(label="Différences Z Normalisées")

# Ajouter les étiquettes et le titre
plt.xlabel('Fréquence spatiale')
plt.ylabel('Amplitude d’aberration')
plt.title('Carte d’erreur LLOWFS Circular Normalisée')
plt.ylim(0, 200)
# Afficher la carte
plt.show()

#%%

# Initialisation des données (exemples)
x = np.linspace(0,h,(nombre_fichiers)-1) # Axe X : 38 valeurs
y = zwfs_amp * 10**9  # Axe Y : 201 valeurs

# tot_diff est déjà un tableau 2D de dimensions (38, 201)
# Exemple : tot_diff = np.random.random((38, 201))  # Supposons que c'est le résultat de tes calculs

# Création de la carte de chaleur
plt.figure(figsize=(8, 6))

# Grille pour les axes
X, Y = np.meshgrid(x, y)  # Créer une grille pour correspondre aux dimensions (38, 201)

# Utiliser contourf pour une carte de chaleur
plt.contourf(X, Y,(tot_diff.T), levels=50, cmap='viridis')  # Remarque : on transpose tot_diff pour correspondre à X et Y
plt.colorbar(label="Différences Z (tot_diff)")

# Ajouter les étiquettes et le titre
plt.xlabel('frequence spatiale')
plt.ylabel('Amplitude d aberration' )
plt.title('Carte derreur ZWFS Circular')

# Afficher la carte
plt.show()

"""

