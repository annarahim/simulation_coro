# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:31:58 2024

@author: arahim
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pathlib import Path
cube_amp=np.arange(-200E-9, 200E-9,((200E-9)*2/100))
valeurs_amp=200E-9
zwfs_amp=cube_amp=np.arange(-valeurs_amp, valeurs_amp,((valeurs_amp)/100)) #pour llowfs pur le reste mettre ca np.arange(-200E-9, 200E-9,((200E-9)*2/100)) : #np.arange(-valeurs_amp, valeurs_amp,((valeurs_amp)/100))
#ZWFS\CIRCULAR\données _circular_sinx
#resultats LLOWFS+ZWFS_polynomes Zernike\response curves\sensor LLOWFS
#ZWFS+LLOWFS\CIRCULAR\VORTEX\données C VORTEX sin x
chemin = Path(r"C:\Users\arahim\Nextcloud\PhD\simu\COMPARAISON ZERNIKE\ZWFS+LLOWFS\SUBARU\FQPM\sans defoc mais FQPM asymetrqiue\x0-10 y0-10")
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

a= np.load(chemin / f'tilt.txt.npy')
b= np.load(chemin / f'tip.txt.npy') 
c= np.load(chemin / f'deoc.txt.npy') 
d= np.load(chemin / f'astig_oblique.txt.npy')
e=np.load(chemin / f'astig_vertical.txt.npy')
f=np.load(chemin / f'trefoil_vertical.txt.npy') 
g= np.load(chemin / f'coma_vertical.txt.npy') 
h=np.load(chemin / f'coma_horizontal.txt.npy')
i=np.load(chemin / f'trefoil_oblique.txt.npy') 
j=np.load(chemin / f'abé_spherique.txt.npy')

x = zwfs_amp * 10**9

#%%
# Calcul des différences
plt.figure()
cube_amp_rad=x
plt.plot(cube_amp_rad,a* 10**9,label='tilt')

plt.plot(cube_amp_rad,b*10**9,label='tip')
plt.plot(cube_amp_rad,c*10**9,label='defocus')
plt.plot(cube_amp_rad,d*10**9,label='astig_oblique')
plt.plot(cube_amp_rad,e*10**9,label='astig_vertical')

plt.plot(cube_amp_rad, f*10**9,label=' trefoil_vertical')
plt.plot(cube_amp_rad,g*10**9,label='coma_vertical')
plt.plot(cube_amp_rad,h*10**9,label='coma_horizontal')
plt.plot(cube_amp_rad,i*10**9,label='trefoil_oblique')
#plt.plot(cube_amp_rad,j*10**9,label='abé_spherique')
plt.plot(cube_amp_rad,cube_amp_rad,label='unitary response')
plt.xlabel('amplitude error applied (nm)')
plt.ylabel('measurement error (nm)')
plt.legend() 
plt.ylim(-200,200)
plt.xlim(-200,200)
plt.title(f' {last_five_folders}')
#%%
#diff1=a*10**9-x
x = zwfs_amp * 10**9
diff_list = [
    b * 10**9 - x,
    c * 10**9 - x,
    d * 10**9 - x,
    e * 10**9 - x,
    f * 10**9 - x,
    g * 10**9 - x,
    h * 10**9 - x,
    i * 10**9 - x,
    j * 10**9 - x
]

  
# Liste des noms descriptifs correspondants
noms_descriptifs = [
    'Astigmatism Oblique',
    'Astigmatism Vertical',
    'Coma Horizontal',
    'Coma Vertical',
    'Deocentering',
    'Tilt',
    'Tip',
    'Trefoil Oblique',
    'Trefoil Vertical'
]

# Définir le seuil pour identifier les dépassements
seuil = 0.5  # Ajustez cette valeur selon vos besoins

# Listes pour stocker les limites en x
limite_negative_x = []
limite_positive_x = []

# Parcourir les différences pour chaque série
nombre_fichiers = len(diff_list)
for h in range(nombre_fichiers):
    tot_diff = diff_list[h]

    # Parcourir jusqu'à la moitié de la longueur de tot_diff
    for i in range(len(tot_diff) // 2):  # On part du centre
        # Prendre les deux points symétriques autour de 0
        val_neg = tot_diff[len(tot_diff) // 2 - i - 1]  # Valeur négative symétrique
        val_pos = tot_diff[len(tot_diff) // 2 + i]      # Valeur positive symétrique

        # Vérifier si les deux valeurs dépassent le seuil
        if abs(val_neg) > seuil or abs(val_pos) > seuil:
            # Ajouter les x correspondants aux différences
            limite_negative_x.append(x[len(tot_diff) // 2 - i - 1])  # x correspondant à la différence négative
            limite_positive_x.append(x[len(tot_diff) // 2 + i])      # x correspondant à la différence positive
            break  # Sortir de la boucle dès qu'une limite est trouvée
plt.figure(2)
# Tracer les différences avec les noms descriptifs
for h in range(nombre_fichiers):
    plt.plot(x, diff_list[h])#, label=noms_descriptifs[h])  # Utiliser les noms descriptifs

# Affichage de la légende et du graphique


# Affichage des limites trouvées
print("Limites négatives en x :", limite_negative_x)
print("Limites positives en x :", limite_positive_x)

#%%    
limite_negative_x_mean=np.mean(limite_negative_x)
limite_positive_x_mean=np.mean(limite_positive_x)
#%%RPZ
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
plt.legend(loc='upper right') 
plt.ylim(-100,100)
plt.title(f' {last_five_folders}')
#np.save(f'differences_specifiques_{last_five_folders}.txt',differences_specifiques) 
#%%
"""
np.save(f'differences_specifiques_{last_five_folders}.txt',differences_specifiques) 
"""
plt.figure()
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