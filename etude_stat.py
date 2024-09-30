# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 13:27:09 2024

@author: arahim
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pathlib import Path
cube_amp=np.arange(-200E-9, 200E-9,((200E-9)*2/100))
valeurs_amp=200E-9
zwfs_amp=cube_amp=np.arange(-200E-9, 200E-9,((200E-9)*2/100))#pour llowfs pur le reste mettre ca : #np.arange(-valeurs_amp, valeurs_amp,((valeurs_amp)/100))


chemin = Path(r"C:\Users\arahim\Nextcloud\PhD\simu\COMPARAISON cycle_pup\LLOWFS\SUBARU\VORTEX\données S vortex sin x")
#nombre eleement dans le dossier (-1)=?"comande à rajouter"
#dernier_element=chemin.name
nombre_fichiers = len([f for f in chemin.glob('*') if f.is_file()])  # Compte  les fichiers

path=chemin

# Récupérer les 5 derniers noms de dossiers
last_five_folders = list(path.parts)[-4:]

# Afficher les résultats
print(last_five_folders)
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

#%%
for i in range(0, (nombre_fichiers)-1, 1):
   
    a1=np.load(chemin / f'test_zwfs{h}.txt.npy') 
     
    x = zwfs_amp[mask]
    y = a1[mask ]
    y_tot.append(y) 
    # Fonction pour calculer le coefficient R^2
    def calc_r_squared(x, y):
        slope, intercept, r_value, _, _ = linregress(x, y)
        return r_value**2

    
    window_size = 4 # Taille minimale de la plage linéaire
    r_squared_threshold = 0.999  # Seuil de qualité du fit 

    
    linear_ranges = []

    # Parcourir toutes les combinaisons possibles de sous-ensembles des données
    for i in range(len(x) - window_size + 1):
        for j in range(i + window_size, len(x) + 1):
            x_subset = x[i:j]
            y_subset = y[i:j]
            
            # Calculer le coefficient de détermination R^2
            r_squared = calc_r_squared(x_subset, y_subset)
            
            # Si le fit est bon, on ajoute cette plage à la liste
            if r_squared >= r_squared_threshold:
                linear_ranges.append((i, j))

    # Sélectionner la plage la plus longue parmi celles trouvées
    if linear_ranges:
        longest_range = max(linear_ranges, key=lambda r: r[1] - r[0])
        x_linear = x[longest_range[0]:longest_range[1]]
        y_linear = y[longest_range[0]:longest_range[1]]


    x_lin_deb.append(x_linear[0])
    x_lin_fin.append(x_linear[-1])
        
    y_list = a1

    x_target_values = [x_linear[0],10E-9,x_linear[-1]]

    
# Boucle sur chaque valeur cible de x
    for i, x_target_value in enumerate(x_target_values):
    # Trouver l'indice de la valeur de x la plus proche de x_target_value
        index = np.argmin(np.abs(x - x_target_value))
    
   
        x_closest_value = x[index]
    
    # Calculer l'écart entre y et x à cette position
        y_value_at_target = y[index]
        x_value_at_target = x[index]
        ecart = y_value_at_target - x_value_at_target
    
    # Stocker l'écart dans le tableau correspondant
        if i == 0:
            ecart_min.append(ecart)
        elif i == 1:
            ecart_midd.append(ecart)
        elif i == 2:
            ecart_max.append(ecart)
        
        
    h += 10
#%%    

 # Calculer l'écart-type des écarts
ecart_type = np.std(ecarts)
ecart_type_min = np.std(ecart_min)
ecart_type_midd = np.std(ecart_midd)
ecart_type_max = np.std(ecart_max)
ecarts_total = np.concatenate([ecart_min, ecart_midd, ecart_max])

# Calculer l'écart-type total
ecart_type_total = np.std(ecarts_total)
mean_deb=np.mean( x_lin_deb)
mean_fin=np.mean( x_lin_fin)

#sigma_fin=
#a1 = np.load(chemin / 'test_zwfs0.txt.npy')
#plt.plot(cube_amp,a1)
#%%# Visualiser les résultats

y_moyen = np.mean(y_tot, axis=0)


plt.scatter(x, y_moyen, label='Données moyenne')

slope, intercept, _, _, _ = linregress(x_linear, y_linear)
y_fit = slope * x_linear + intercept
plt.plot(x_linear, y_fit, color='r', label='Fit linéaire')
plt.scatter(x_linear, y_linear, color='g', label=f'Plage linéaire[ {mean_deb:.2e} ; {mean_fin:.2e}] ')
plt.legend(title=f'Écarts-types\nMin: {ecart_type_min:.3e}\nMidd: {ecart_type_midd:.3e}\nMax: {ecart_type_max:.3e}\nTotal: {ecart_type_total:.3e}')
plt.xlabel('Amplitude of cycl/pup applied (m)')
plt.ylabel('Amplitude of cycl/pup measured (m)')
plt.grid()
plt.title(f' {last_five_folders}')
plt.show()
