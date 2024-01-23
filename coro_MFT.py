# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 14:12:37 2024

@author: arahim
"""

import numpy as np
import matplotlib.pyplot as plt

#%%Definition des variables 
wavelength = 700E-6  # Longueur d'onde de la lumière en metres
D=8 #Diamètre de l'ouverture circulaire en m 
NA = 750 # Number of pixels along the pupil diameter D
NB = 100 # Number of pixels in the focal plane
NC=NA #Nombre de pixels au plan C
ND=1024 #Nombre de pixels au plan D
md =  20 #echantillonage pour la TF au plan d 
eps=2 #coefficcient epsilon eps=1 pour coro Lyot et eps=2 pour coro Rodier
m =  1.22*2 #echantillonage en (lambda/D)

#%%Creation de l'ouverture d'entrée 
circle_radius = NA/2 #rayon de l'ouverture en nmr de pixel
x0=(NA/2)
y0=(NA/2)

# Création d'une grille de points dans le plan XY
x1 = np.linspace(0, NA-1, NA)
y1 = np.linspace(0, NA-1, NA)
x, y = np.meshgrid(x1, y1)


#Création de l'ouverture circulaire 
aperture_radius = (x-x0)**2 + (y-y0)**2 <= circle_radius**2
P=np.zeros((NA,NA))
P[aperture_radius]=1

#%%Creation du masque du corono 
circle_radius_MFP = (NB/2) #rayon du masque focal MFP
x0_MFP=(NB/2)
y0_MFP=(NB/2)

# Création d'une grille de points dans le plan XY
x1_MFP = np.linspace(0, NB-1, NB)
y1_MFP = np.linspace(0, NB-1, NB)
x_MFP, y_MFP = np.meshgrid(x1_MFP, y1_MFP)

#Création de l'ouverture circulaire 
aperture_radius = (x_MFP-x0_MFP)**2 + (y_MFP-y0_MFP)**2 <= circle_radius_MFP**2
MFP=np.zeros((NB,NB))
MFP[aperture_radius]=1

plt.figure('m')
plt.title('m')
plt.imshow(MFP)
plt.colorbar(label='Intensité')

#%%Creation du lyot stop 
LS=(circle_radius*1)# Rayon du lyot stop en fonction du rayon de l'ouverture d'entrée 
radius2 = (x-x0)**2 + (y-y0)**2 <= LS**2
P_C=np.zeros((NA,NA))
P_C[radius2]=1
plt.figure('LS')
plt.title('LS')
plt.imshow(P_C)
plt.colorbar(label='Intensité')



#%%Fonction calculant la Transformée de Fourier semi analytique 

def MFT (NA,NB,m,E,inv=False):
    """
    NA = tableau d'entrée
    NB= tableau de sortie
    m = echantillonage en lamda/D
    E=le champ electrique du plan précedent
    inv= False par defaut => signe =-1 pour une TF et inv= True => signe=1 pour une TF inverse 
    """
    #Definition des vecteurs pour la TF 
    xk = yk = (np.arange(NA) - NA/2)  * 1/NA
    ul = vl = (np.arange(NB) - NB/2) * m/NB
    #mise en forme des vecteur sous forme de colonne
    U = (np.array(ul)).reshape(-1, 1)
    X = (np.array(xk)).reshape(-1, 1)
    V = (np.array(ul)).reshape(-1, 1)
    Y = (np.array(xk)).reshape(-1, 1)    
    if inv == False:
        signe=-1
    else :
        signe=1
    #F_U = calcul de la TF 1D de f(x)
    F_U = (np.exp(signe*2j * np.pi * U @ X.T))

    #F_V = calcul de la TF 1D de f(y)
    F_V = (np.exp(signe*2j * np.pi *  Y @ V.T))


    #Fb_UH = TF 2D normaliser par m/NA.NB et donc multiplier par la matrice f(x,y)=E
    Fb_UH  = (m / (NA * NB)) *F_U @E @ F_V
    
    return Fb_UH

#%% PLAN A
E_A=1*P # champ electrique plan A 
I_A=np.abs(E_A)**2 #Intensité plan A 
plt.figure('ouverture')
plt.title('ouverture')
plt.imshow(P)
plt.colorbar(label='Intensité')
plt.show()
 

#%% PLAN B 
#Calcul de la TF direct du champ elec au plan B 
E_B = MFT(NA, NB, m, E_A)
I_B=np.abs(E_B )**2
plt.figure('plan B avant masque')
plt.title('plan B avant masque')
plt.imshow(np.log10(I_B))
plt.colorbar(label='Intensité')   
E_B2=E_B*MFP #Multiplication du champ B par le masque 
I_B2=np.abs(E_B2)**2
plt.figure('plan B apres MFP')
plt.title('plan B apres MFP')
plt.imshow((I_B2)**0.25) #puissance 0.25 pour un meilleur affichage
plt.colorbar(label='Intensit"é')

#%% PLAN C
#Calcul de la TF inverse du champ elec du plan B  
E_C= MFT(NB, NA, m, E_B2,inv=True)
I_C=np.abs(E_C)**2
plt.figure('c')
plt.title('c')
plt.imshow((I_C)**0.25)
plt.colorbar(label='Intensité')

#Determination du champ elec au plan C multiplier par le lyot stop 
E_C2=(E_A-eps*E_C)*(P_C)
I_C2=np.abs(E_C2)**2
plt.figure('plan C apres lyot stop')
plt.title('plan C apres lyot stop')
plt.imshow(I_C2)
plt.colorbar(label='Intensité')

plt.figure()
plt.title('coupe de I_C')
plt.ylabel('I_C')
frequence = np.fft.fftfreq((NC),(NC*wavelength*m/(2*D)))
frequence2=np.fft.fftshift(frequence)
positive_frequencies2 = frequence2[int(NC):]
coupes2=I_C2[NC//2,:NC]
plt.plot(frequence2,coupes2, color='blue', alpha=1,label='plan b')

#%% PLAN D 
#Calcul de la TF du champ elec au plan C donnant le champ elec au plan D 
E_D = MFT(NC,ND,md,E_C2)
I_D=np.abs(E_D)**2
plt.figure('PLAN D')
plt.title('plan D')
plt.imshow((I_D)**0.25)
plt.colorbar(label='Intensité')

#Determination du champ elec sans corono
E_C0=E_A*P_C
E_D0=MFT(NC,ND,md,E_C0)
I_D0=np.abs(E_D0)**2
plt.figure('PLAN D0')
plt.title('plan D0')
plt.imshow((I_D0)**0.25)
plt.colorbar(label='Intensité')

#Normalisation de l'intensité au plan D selon une propagation sans coronographe 
# Trouver le maximum dans la liste 
diviseur =max(max(row) for row in I_D0)
#I_Dnorm=I_D/diviseur
I_Dnorm=I_D/diviseur
plt.figure('PLAN D normalisé avec le max')
plt.title('plan D normalisé avec le max')
plt.imshow(I_Dnorm)
plt.colorbar(label='Intensité')
I_D0norm=I_D0/diviseur


# TF de la frequence sur l'echantillon ND par pas de lambda/D
freq = np.fft.fftfreq((ND),(ND*wavelength/(2*D)))
freq2=np.fft.fftshift(freq)
positive_frequencies = freq2[int(ND//2):] #selection des frequences positives
#Coupes selon un axe de l'intensité au plan D 
plt.figure()
plt.plot(positive_frequencies,np.log10(I_D0norm[ND//2,ND//2:ND]) , color='blue', alpha=1,label='propagation sans corono')
plt.plot(positive_frequencies,np.log10(I_Dnorm[ND//2,ND//2:ND]), color='red', alpha=1,label='propagation avec corono')
plt.legend()
plt.title('Coupes radiales')
plt.xlabel('Lambda/D')
plt.ylabel('I_D')
plt.show()



