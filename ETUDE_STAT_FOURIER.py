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
zwfs_amp=cube_amp=np.arange(-valeurs_amp, valeurs_amp,((valeurs_amp)/10)) #pour llowfs pur le reste mettre ca np.arange(-200E-9, 200E-9,((200E-9)*2/100)) : #np.arange(-valeurs_amp, valeurs_amp,((valeurs_amp)/100))
#ZWFS\CIRCULAR\données _circular_sinx
#resultats LLOWFS+ZWFS_polynomes Zernike\response curves\sensor LLOWFS
#ZWFS+LLOWFS\CIRCULAR\VORTEX\données C VORTEX sin x
chemin = Path(r"C:\Users\arahim\Nextcloud\PhD\simu\COMPARAISON cycle_pup\ZWFS+LLOWFS\SUBARU\FQPM\avec defoc")
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
h=0
for i in range(1, (nombre_fichiers)-1, 1):
   
    a1=np.load(chemin / f'test_zwfs{h}.txt.npy') 
     
    x = zwfs_amp*10**9#[mask]
    y = a1*10**9#[mask ]
    y_tot.append(y)
    differences = [a - b for a, b in zip(y, x)]
    tot_diff.append(differences)
    
    plt.plot(x,differences,label = f"{h}")
    #plt.legend()
    #plt.legend(fontsize=8) 

    # Recherche des limites
    for i in range(len(differences)//2):  # On part du centre
        # Prendre les deux points symétriques autour de 0 (un positif, un négatif)
        val_neg = differences[len(differences)//2 - i - 1]  # Valeur négative symétrique
        val_pos = differences[len(differences)//2 + i]      # Valeur positive symétrique
        
        # Vérifier si les deux valeurs sont inférieures au seuil
        if abs(val_neg) > seuil or abs(val_pos) > seuil:
            # Si l'une des deux valeurs dépasse le seuil, on s'arrête
            limite_negative_x .append( x[len(differences)//2 - i - 1])  # x correspondant à la différence négative
            limite_positive_x.append( x[len(differences)//2 + i]  )    # x correspondant à la différence positive
            break
    for target in x_target_values:
        # Trouver l'indice de la valeur de x la plus proche de target
        index = np.argmin(np.abs(x - target))
        differences_specifiques.append(differences[index]) 
    """
    y_tot.append(y) 
    # Fonction pour calculer le coefficient R^2
    def calc_r_squared(x, y):
        slope, intercept, r_value, _, _ = linregress(x, y)
        return r_value**2

    
    window_size = 4 # Taille minimale de la plage linéaire
    r_squared_threshold = 0.9999  # Seuil de qualité du fit 

    
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
        """
    freq_spatial.append(h)    
 
    h += 1
#%%
#plt.ylim(-45,45)
#plt.yticks([-40,-20,0,20,40])
plt.ylim(-200,200)
plt.yticks([-200,-150,-100,-50,0,50,100,150,200])
plt.xticks([-200,-150,-100,-50,0,50,100,150,200])
plt.plot(x,x,'--',label='unitary response')
plt.legend(fontsize=6)#loc='lower right')
#plt.xlim(-200,200)
plt.title("ZWFS CIRCULAR ")
plt.xlabel('Error applied (nm)')
plt.ylabel('Error measured (nm)')
#%%    
limite_negative_x_mean=np.mean(limite_negative_x)
limite_positive_x_mean=np.mean(limite_positive_x)
#%%RPZ
"""
y_moyen = np.mean(tot_diff, axis=0)
y_plot=y_moyen*10**9
x_plot=x*10**9
#plt.figure()
#plt.scatter(x, tot_diff)
#plt.scatter(x_plot,y_plot )
plt.axhline(y=0.5, color='r', linestyle='--', label="seuil = +/- 0,5 nm")
plt.axhline(y=-0.5, color='r', linestyle='--')
plt.axvline(x=limite_negative_x_mean, color='b', linestyle='--' ,label=f"Intervalle linéaire [ {limite_negative_x_mean:.2e} ; {limite_positive_x_mean:.2e}]") #label=f"Intervalle linéaire [ {limite_negative_x_mean:.2e} ; {limite_positive_x_mean:.2e}]")
plt.axvline(x=limite_positive_x_mean, color='b', linestyle='--')
plt.xlabel('amplitude error applied (nm)')
plt.ylabel('measurement error (nm)')
plt.legend() 
plt.title(f' {last_five_folders}')
#np.save(f'differences_specifiques_{last_five_folders}.txt',differences_specifiques) 
"""
#%%

"""
np.save(f'differences_specifiques_{last_five_folders}.txt',differences_specifiques) 
"""
#plt.figure()
#plt.plot(freq_spatial,differences_specifiques)
#plt.ylim(-5.5E-9,-5E-9)
#%%
"""
r1=np.load("differences_specifiques_['LLOWFS', 'CIRCULAR', 'VORTEX', 'donnes sin x'].txt.npy")
r2=np.load("differences_specifiques_['COMPARAISON cycle_pup', 'ZWFS', 'CIRCULAR', 'données _circular_sinx'].txt.npy")
r3=np.load("differences_specifiques_['ZWFS+LLOWFS', 'CIRCULAR', 'VORTEX', 'données C VORTEX sin x'].txt.npy")
plt.plot(freq_spatial,r1,label='LLOWFS')
plt.plot(freq_spatial,r2,label='ZWFS')
plt.plot(freq_spatial,r3,label='ZWFS+LLOWFS')
plt.xlabel('Frequence spatiale')
#plt.xlim(0,240)
plt.ylabel('measurement error (nm)')
plt.title(' comparaison eerreur de mesure CIRCULAR VORTEX x=200 nm')
plt.legend()
"""
#%%
"""
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
"""