# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 13:28:04 2024

@author: arahim
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math 

from pathlib import Path

#%% VARIABLE DEFINTIONS
wavelength = 1.625E-6 # wavelenght in meters
D_tel=8 #Diameter of telescopes in meters
D=300# Diameter of circular aperture in meters
N=512 #Number of pixel for zerro padding 
NA = D# Number of pixels along the pupil diameter D
NB=50# Number of pixels in the focal plane
NC=NA #Number of pixels in plane C
ND=75#Number of pixels in plane D
m1 =1.06#sampling (wavelenght/D)
#md =20# 20 sampling for TF in plane D 
D_win=50
phase=np.pi/2
lp=4
oversize=3
mode_mesure=3
rayon_ob_central=(D / 2)/ 100


#%%Creation de l'ouverture d'entrée with zero padding 

def create_aperture(N, D, type='circular', epaisseur_spyder=1):
    """
    Crée une ouverture dans une grille NxN.

    Paramètres:
    N : int - Taille de la grille.
    D : float - Diamètre de l'ouverture.
    type : str - Type d'ouverture ('circular' ou 'subaru').
    epaisseur_spyder : int - Épaisseur des bras de la croix pour l'ouverture Subaru.

    Retour:
    P : np.array - Matrice représentant l'ouverture.
    """
    x0 = y0 = (N / 2)

    # Création de la grille de points
    x1 = np.linspace(0 + 0.5, N - 1 + 0.5, N)
    y1 = np.linspace(0 + 0.5, N - 1 + 0.5, N)
    x, y = np.meshgrid(x1, y1)

    # Création d'une grille de 0 de taille NxN
    P = np.zeros((N, N))

    if type == 'circular':
        # Création de l'ouverture circulaire
        circle_radius = D / 2
        aperture_radius = (x - x0) ** 2 + (y - y0) ** 2 <= circle_radius ** 2
        P[aperture_radius] = 1

    elif type == 'subaru':
        # Création de l'ouverture Subaru avec obstruction centrale et croisillons
        circle_radius = D / 2
        aperture_radius = (x - x0) ** 2 + (y - y0) ** 2 <= circle_radius ** 2
        P[aperture_radius] = 1

        # Création de l'obstruction centrale
        circle_radius2 = rayon_ob_central# (7.92 / 2.3))
        central_obstruction = (x - x0) ** 2 + (y - y0) ** 2 <= circle_radius2 ** 2
        P[central_obstruction] = 0

        # Création des croisillons (spyder)
        for i in range(N):
            for j in range(N):
                if aperture_radius[i, j]:
                    if abs(i - j) < epaisseur_spyder or abs(i + j - (N - 1)) < epaisseur_spyder:
                        P[i, j] = 0

    return P
#%%RLS
def create_diameter_limit(N, D, oversize, type='circular', epaisseur_spyder=1):
    """
    Crée une limite de diamètre pour la récupération de la lumière après LS.

    Paramètres:
    N : int - Taille de la grille.
    D : float - Diamètre de l'ouverture.
    oversize : float - Facteur d'oversize pour l'ouverture.
    type : str - Type d'ouverture ('circular' ou 'subaru').
    epaisseur_spyder : int - Épaisseur des bras de la croix pour l'ouverture Subaru.

    Retour:
    D_lim_LS : np.array - Matrice représentant la limite de diamètre.
    """
    x03=0
    y03=0

    # Grille de points dans le plan XY pour le diamètre limite
    x13 = (np.arange(N) - (N / 2) + 0.5) / (oversize * D / 2)
    y13 = (np.arange(N) - (N / 2) + 0.5) / (oversize * D / 2)
    x3, y3 = np.meshgrid(x13, y13)
    x0=((N/2))#-(1/2) #centrage du cercle au milieu de la grille de taille N
    y0=((N/2))
    x1 = np.linspace(0+0.5, N-1+0.5, N)
    y1 = np.linspace(0+0.5, N-1+0.5, N)
    x, y = np.meshgrid(x1, y1)

    # Création d'une grille de 0 de taille NxN
    D_lim_LS = np.zeros((N, N))

    # Création de l'ouverture pour la récupération de lumière après LS
    circle_radius3 = 1  # Radius de l'ouverture par rapport à la taille du tableau
    aperture_radius3 =  (x3-x03)**2 + (y3-y03)**2 <= circle_radius3**2
    D_lim_LS[aperture_radius3] = 1

    if type == 'circular':
        # Création d'une pupille extérieure circulaire
        circle_radius_outer_pupill = D / 2
        aperture_radius_outer_pupill = (x-x0)**2 + (y-y0)**2 <= circle_radius_outer_pupill**2
        D_lim_LS[aperture_radius_outer_pupill] = 0

    elif type == 'subaru':
        # Création d'une pupille extérieure circulaire avec croisillons
        circle_radius_outer_pupill = D / 2
        aperture_radius_outer_pupill = (x-x0)**2 + (y-y0)**2 <= circle_radius_outer_pupill**2
        D_lim_LS[aperture_radius_outer_pupill] = 0

        # Taille de l'obstruction centrale
        taille = int(1.2*rayon_ob_central )#(7.92 / 2.3))))
        x_start = (N) // 2
        y_start = (N) // 2
        CAR_bis = np.zeros((N, N))
        ini6 = x_start - taille
        end6 = y_start + taille
        CAR_bis[ini6:end6, ini6:end6] = 1
        D_lim_LS[CAR_bis == 1] = 1

        # Ajout des croisillons (spyder)
        epaisseur_D_lim = epaisseur_spyder * 1.5
        for i in range(N):
            for j in range(N):
                if aperture_radius_outer_pupill[i, j]:
                    if abs(i - j) < epaisseur_D_lim or abs(i + j - (N - 1)) < epaisseur_D_lim:
                        D_lim_LS[i, j] = 1

    return D_lim_LS


#%%Creation focal mask ZWFS
circle_radius_MFP_zwfs = NB//2  #rayon du masque focal MFP
x0_MFP_zwfs=NB//2
y0_MFP_zwfs=NB//2 

# Création d'une grille de points dans le plan XY
x1_MFP_zwfs = np.linspace(0+0.5, NB-1+0.5, NB)
y1_MFP_zwfs = np.linspace(0+0.5, NB-1+0.5, NB)
x_MFP_zwfs, y_MFP_zwfs = np.meshgrid(x1_MFP_zwfs, y1_MFP_zwfs)
    
#Création de l'ouverture circulaire 

aperture_radius2_zwfs = (x_MFP_zwfs-x0_MFP_zwfs)**2 + (y_MFP_zwfs-y0_MFP_zwfs)**2 <= circle_radius_MFP_zwfs**2
MFP_zwfs=np.zeros((NB,NB))
MFP_zwfs[aperture_radius2_zwfs]=1

"""
plt.figure('m')
plt.title('m')
plt.imshow(MFP_zwfs-np.fliplr(MFP_zwfs))
plt.colorbar(label='Intensité')

"""
#%%

X4 = (np.arange(N)-(N/2))/(N/2)
Y4 = (np.arange(N)-(N/2))/(N/2)
x44, y44 = np.meshgrid(X4, Y4)
rho4=np.sqrt(x44**2 + y44**2) 
theta4=np.arctan2(y44,x44)

#%%
circle_radius1 = 1 #diameter of aperture par rapport à la taille du tableau
x01=0
y01=0

# creation a point grid in the xy plane
x11 = (np.arange(D)-(D/2)+0.5)/(D/2)
y11 = (np.arange(D)-(D/2)+0.5)/(D/2)
x111, y111= np.meshgrid(x11, y11)
rho1 = np.sqrt(x111**2 + y111**2) 
theta1 = np.arctan2(y111,x111)
def zernike_polar(theta0,rho0, n, m):
    
    if m == 0:
        return  zernike_radial(n, m, rho0) * np.sqrt(n + 1)
    elif m > 0:
        return  zernike_radial(n, m, rho0)  * np.sqrt(2 * (n + 1))* np.cos(m * theta0)
    else:
        return  zernike_radial(n, abs(m), rho0) * np.sqrt(2 * (n + 1)) * np.sin(abs(m) * theta0)

def zernike_radial(n, m ,rho0):
    
    result=np.zeros_like(rho0)
    for k in range((n - abs(m)) // 2 + 1):
        result += ((-1) ** k * np.math.factorial(n - k) / (np.math.factorial(k) * np.math.factorial((n + abs(m)) // 2 - k) * np.math.factorial((n - abs(m)) // 2 - k) ) * rho0 ** (n - 2 * k))
    result[rho0>1]=0
    return result



#%% MFT
def MFT (NA,NB,m1,E,inv=False, center_pixel=False):
    """
    NA = Input table
    NB= output tabe
    m = sampling in lamda/D
    E=Electric field of precedent plane
    inv= False by default => sign =-1 for do a TF and inv= True => sign=1 for a invers TF 
    """
    if center_pixel==False:
        val=0
    else :
        val=1/2
    #Vectors defintinion for TF 
    xk = (np.arange(NA) - NA/2+val)  * 1/NA 
    ul = (np.arange(NB) - NB/2+ val ) * m1/NB 
    #vector formatting in column format
    U = (np.array(ul)).reshape(-1, 1)
    X = (np.array(xk)).reshape(-1, 1)
    V = (np.array(ul)).reshape(-1, 1)
    Y = (np.array(xk)).reshape(-1, 1)    
    if inv == False:
        sign=-1
    else :
        sign=1
    #F_U = calcul of TF 1D  of f(x)
    F_U = (np.exp(sign*2j * np.pi * U @ X.T))

    #F_V = calcul of TF 1D of f(y)
    F_V = (np.exp(sign*2j * np.pi *  Y @ V.T))


    #Fb_UH = TF 2D normalized by m/NA.NB and  multiply by matrix: f(x,y)=E
    Fb_UH  = (m1 / (NA * NB)) *F_U @E @ F_V
    
    return Fb_UH
#%%

# Fonction pour créer le masque à quatre quadrants
def create_quadrant_mask(N):
    x0_MFP = (((N)/2))#-(2)#+2#-(1/2)
    y0_MFP = (((N)/2))#-(1/2)

    # Création d'une grille de points dans le plan XY
    x1_MFP = np.linspace(0, N-1, N)
    y1_MFP = np.linspace(0, N-1 ,N)
    x_MFP, y_MFP = np.meshgrid(x1_MFP, y1_MFP)

    # Création de l'ouverture circulaire
    # Création du masque à quatre quadrants dans l'ouverture circulaire


    # Appliquer le masque à quatre quadrants
    MFP = np.ones((N, N))#◘,dtype="complex128")
    MFP[(x_MFP < x0_MFP) & (y_MFP < y0_MFP)] =- 1  # Quadrant supérieur gauche
    MFP[(x_MFP >= x0_MFP) & (y_MFP < y0_MFP)] =1 # Quadrant supérieur droit
    MFP[(x_MFP < x0_MFP) & (y_MFP >= y0_MFP)] = 1# Quadrant inférieur gauche
    MFP[(x_MFP >= x0_MFP) & (y_MFP >= y0_MFP)] = -1  # Quadrant inférieur droit

    return MFP

# Fonction pour créer le masque vortex
def create_vortex_mask(N, D, lp):
    MFP_temp = np.exp(1j * lp * theta1)
    MFP = np.zeros((N, N), dtype="complex128")
    ini = (N // 2) - (D // 2)
    end = (N // 2) + (D // 2)
    MFP[ini:end,ini:end]= MFP_temp
    
    return MFP

# Fonction principale pour choisir et créer le masque
def create_mask(N, mask_type, D=None, lp=None):
    if mask_type == 'quadrant':
        return create_quadrant_mask(N)
    elif mask_type == 'vortex':
        if D is None or lp is None:
            raise ValueError("D et lp doivent être spécifiés pour le masque vortex")
        return create_vortex_mask(N, D, lp)
    else:
        raise ValueError("Type de masque non valide. Utilisez 'quadrant' ou 'vortex'.")



#%%CHOIX ouverture/mask
# Choisir le type de masque
mask_type = 'quadrant'  # ou 'quadrant'

# Créer le masque
MFP = create_mask(N, mask_type,D,lp)



aperture='subaru'
# Création de la limite de diamètre pour une ouverture circulaire simple
D_lim_LS = create_diameter_limit(N, D, oversize, type=aperture)


P = create_aperture(N, D, type=aperture)
start = (N - D) // 2
O = P[start:start + D, start:start + D]

plt.imshow((MFP)-np.flipud (-MFP))




#%% Determination of electric fiel without aberartions
amp_defoc=1.76E-6
modes0=[(2,0)]#1.27E-6#1.8E-6
vecteurs_I0=[]
opd_def=np.zeros((D,D))
for  mode0 in (modes0):
    modes0=0
    n0, m0 = mode0
    #pour le tip tilt amplitude = amplitude_rad/(2*np.pi)
    zernike_mode4 = zernike_polar( theta4,rho4, n0, m0)
    opd_def=amp_defoc*zernike_mode4
#plt.imshow(opd_def)
E_A0_temp=1*O#Calcul of E_A temporary in the aperture O
E_A0=np.zeros((N,N),dtype="complex128")
ini=(N//2)-(D//2)
end=(N//2)+(D//2)
E_A0[ini:end,ini:end]=E_A0_temp
#Injection of E_A in a grill with zero padding



E_B0_before_mask=MFT(N,N, N/2, E_A0,center_pixel=True)
E_B0_after_mask=E_B0_before_mask*MFP 

E_C0_before_LS=MFT(N, N, N/2,E_B0_after_mask,inv=True,center_pixel=True)
E_C0_after_LS=(E_C0_before_LS)*D_lim_LS#*np.exp((2*1j*np.pi*opd_def)/wavelength)


E_D0=MFT(N, N, N/2,E_C0_after_LS,center_pixel=True)
I_D0=np.abs(E_D0)**2


ini1=(N//2)-(D_win//2)
end1=(N//2)+(D_win//2)
I_D0_bis=I_D0[ini1:end1,ini1:end1]
 
vector_I_D0 = I_D0_bis.flatten() #colonne vecteur 1D 
vecteurs_I0.append(vector_I_D0) #liste du vecteur 1D

#%%ref pour ZWFS

E_D0_bis=np.zeros((NB,NB),dtype="complex128")
ini5=(N//2)-(NB//2)
end5=(N//2)+(NB//2)

E_D0_bis=E_D0[ini5:end5,ini5:end5]



E_ref_after_mask_zwfs=E_D0_bis*(MFP_zwfs)


E_C_ref=MFT(NB, N, m1*(N/D),E_ref_after_mask_zwfs,inv=True,center_pixel=True)

E_ref_zwfs=E_C0_after_LS-((1-np.exp(1j*phase))* E_C_ref)
I_ref_temp=(np.abs(E_ref_zwfs)**2).flatten()

#I_ref=I_ref_temp[O==1]




#%% calibartion et commande ZWFS

amp_calib=[20E-9, -20E-9]
zwfs_modes=[(1,-1),(1,1),(2,-2),(2,0),(2,2),(3,-3),(3,-1),(3,1),(3,3),(4,0)]

tab=[]
images=[]
OPD_calib=[]

#%%
for i, ce in enumerate(amp_calib):
    for u, zwfs_mode in enumerate (zwfs_modes):
        n1, m11 =zwfs_mode
        
            
            # Utiliser une valeur différente de cube_amp
        
        #pour le tip tilt amplitude = amplitude_rad/(2*np.pi)
        zernike_mode= (zernike_polar(theta1,rho1, n1, m11))
        #zernike_mode_zwfs0.append(zernike_polar(theta1,rho1, n1, m11))
        OPD=zernike_mode*amp_calib[i] 
        OPD_calib.append(OPD)  

#%%
for q, zern in enumerate (OPD_calib):
    
    E_A_zwfs_temp=1*O*np.exp((1j*2*np.pi*OPD_calib[q])/wavelength) # POLYNOMES DE ZERNIKE *OPD_calib[q]
    #E_A_zwfs_temp=1*O*np.exp((2*np.pi*1j*abe_fourier_zwfs[q])/wavelength)# #ABE FOURIER
    E_A_zwfs=np.zeros((N,N),dtype="complex128")
    ini=(N//2)-(D//2)
    end=(N//2)+(D//2)
    E_A_zwfs[ini:end,ini:end]=E_A_zwfs_temp
    
    E_before_mask_calib_zwfs=MFT(N,N, N/2, E_A_zwfs,center_pixel=True)
    E_after_mask_calib_zwfs=E_before_mask_calib_zwfs*MFP 

    E_before_LS_calib_zwfs=MFT(N, N, N/2,E_after_mask_calib_zwfs,inv=True,center_pixel=True)
    E_after_LS_calib_zwfs=(E_before_LS_calib_zwfs)*D_lim_LS#*np.exp((2*1j*np.pi*opd_def)/wavelength)


    E_D_calib_zwfs=MFT(N, N, N/2,E_after_LS_calib_zwfs,center_pixel=True) 
    
    E_D_calib_zwfs_bis=np.zeros((NB,NB),dtype="complex128")
    ini5=(N//2)-(NB//2)
    end5=(N//2)+(NB//2)

    E_D_calib_zwfs_bis=E_D_calib_zwfs[ini5:end5,ini5:end5]

    E_B_after_mask_calib_zwfs=E_D_calib_zwfs_bis*(MFP_zwfs)   #Multiplying the B electric field by the mask 
    E_C0_calib_zwfs= MFT(NB, N, m1*(N/D),E_B_after_mask_calib_zwfs,inv=True,center_pixel=True)
       
    # PLANE C
    #Calcul of the inverse TF of the elec field in plane B  
    E_C_zwfs= E_after_LS_calib_zwfs-(1-np.exp(1j*phase))* E_C0_calib_zwfs
    #Determination of the elec field at the C plane multiplied by the lyot stop 

 
      # PLANE D 
    #Calcul of the TF of the elec field in plane C giving the elec field in plane D 
    
    OPD_mesure=(np.abs(E_C_zwfs)**2)#*(wavelength/(2*np.pi))
       
        
    image=OPD_mesure
    
    OPD_mesure_bis=OPD_mesure.flatten()
    tab.append(OPD_mesure_bis)
    images.append(OPD_mesure)

#%%
sum1=[]
sum2=[]
images_sum=[]
images_sum2=[]
taille=int(len(tab)/2)
for indice in range(taille):
    
    sum1.append(((tab[indice])-tab[indice+len(zwfs_modes)])/((2*amp_calib[0])))
    images_sum.append(((images[indice])-images[indice+len(zwfs_modes)])/((2*amp_calib[0])))




#%%
vecteur_zwfs = np.array(sum1)#sum1 pour zernike
s_zwfs=vecteur_zwfs.T
control_matrix_zwfs = np.linalg.pinv(s_zwfs) 

#%% POUR TESTER TOUS LES MODES EN MEME TEMPS
valeurs_amp=200E-9
zwfs_amp=np.arange(-valeurs_amp, valeurs_amp,((valeurs_amp)/100))#range(1)
modes_zwfs=[(1,-1),(1,1),(2,-2),(2,0),(2,2),(3,-3),(3,-1),(3,1),(3,3),(4,0)]
step_cubephase_zwfs=[]
zernike_mode2=[]
vecteurs_I_R2=[]
opd=[]
for mode_zwfs in (modes_zwfs):
    n2, m2 = mode_zwfs
    zernike_mode2 = (zernike_polar(theta1,rho1, n2, m2))
    
    # Pour chaque mode, itérer sur toutes les valeurs de cube_amp
    for i in range(len(zwfs_amp)):  # Appliquer la valeur actuelle de cube_amp
        opd_mod = zwfs_amp[i] * zernike_mode2  # Calculer l'opd pour ce mode et cette amplitude
        
        # Stocker l'OPD pour ce mode et cette amplitude dans la liste
        opd.append(opd_mod)    
        E_A1_temp_zwfs=1*O*np.exp((1j*2*np.pi*opd_mod)/wavelength) 
        #Injection of E_A in a grill with zero padding
        E_A1=np.zeros((N,N),dtype="complex128")
        ini=(N//2)-(D//2)
        end=(N//2)+(D//2)
        E_A1[ini:end,ini:end]=E_A1_temp_zwfs      
        
        
        E_B1_before_mask=MFT(N, N, N/2, E_A1,center_pixel=True)
    
        E_B1_after_mask=E_B1_before_mask*MFP 
    
        E_C1_before_LS=MFT(N, N, N/2,E_B1_after_mask,inv=True,center_pixel=True)
    
        E_C1_after_LS=(E_C1_before_LS)*D_lim_LS#*np.exp((1j*2*np.pi*opd_def)/wavelength)
    
        E_R2=MFT(N, N, N/2,E_C1_after_LS,center_pixel=True)
    
        I_R2=np.abs(E_R2)**2
        I_R2_bis=I_R2[ini1:end1,ini1:end1]
        vector_I_R2 = I_R2_bis.flatten()
        vecteurs_I_R2.append(vector_I_R2)
        #step=((control_matrix_C@((vector_I_R2-vector_I_D0))))
        #step_cubephase.append(step*(amp))
    
    
        E_R2_bis=np.zeros((NB,NB),dtype="complex128")
        ini5=(N//2)-(NB//2)
        end5=(N//2)+(NB//2)
    
        E_R2_bis=E_R2[ini5:end5,ini5:end5]
    
        E_R2_after_mask_zwfs=E_R2_bis*(MFP_zwfs)
    
        E_R2_zwfs=MFT(NB, N, m1*(N/D),E_R2_after_mask_zwfs,inv=True,center_pixel=True)
    
        E_R2_zwfs_sensor=E_C1_after_LS-((1-np.exp(1j*phase))* E_R2_zwfs)
        
        OPD_mesure_1bis=(np.abs(E_R2_zwfs_sensor)**2 )#*(wavelength/(2*np.pi))
    
    
        OPD_mesure1=OPD_mesure_1bis.flatten()#[O==1]
        step_zwfs=((control_matrix_zwfs@((OPD_mesure1-I_ref_temp))))
    
        step_cubephase_zwfs.append((step_zwfs))
        
#%%
tilt=[]
tip=[]
astig_oblique=[]
defoc=[]
astig_vertical=[]
trefoil_vertical=[]
coma_vertical=[]
coma_horizontal=[]
trefoil_oblique=[]
abé_spherique=[]
step_cubephase = np.array(step_cubephase_zwfs)
#%%
# Nombre total de valeurs par mode
# Itérer sur le nombre de modes (10 dans ce cas)
for mode_index in range(step_cubephase.shape[1]):  # step_cubephase a 10 colonnes
    for i in range(201):  # 201 valeurs pour chaque mode
        # Indices à récupérer
        value = step_cubephase[mode_index * 201 + i][mode_index]
        
        # Ajouter la valeur à la bonne liste
        if mode_index == 0:
            tilt.append(value)
        elif mode_index == 1:
            tip.append(value)
        elif mode_index == 2:
            astig_oblique.append(value)
        elif mode_index == 3:
            defoc.append(value)
        elif mode_index == 4:
            astig_vertical.append(value)
        elif mode_index == 5:
            trefoil_vertical.append(value)
        elif mode_index == 6:
            coma_vertical.append(value)
        elif mode_index == 7:
            coma_horizontal.append(value)
        elif mode_index == 8:
            trefoil_oblique.append(value)
        elif mode_index == 9:
            abé_spherique.append(value)


#%%

    
cube_amp_rad=(zwfs_amp)#*((2*np.pi)/(wavelength))
plt.figure()
#plt.plot(cube_amp,cycle_etude)
plt.plot(cube_amp_rad,tilt,label='tilt')
plt.plot(cube_amp_rad,tip,label='tip')
plt.plot(cube_amp_rad,defoc,label='defocus')
plt.plot(cube_amp_rad,astig_oblique,label='astig_oblique')
plt.plot(cube_amp_rad,astig_vertical,label='astig_vertical')

plt.plot(cube_amp_rad, trefoil_vertical,label=' trefoil_vertical')
plt.plot(cube_amp_rad,coma_vertical,label='coma_vertical')
plt.plot(cube_amp_rad,coma_horizontal,label='coma_horizontal')
plt.plot(cube_amp_rad,trefoil_oblique,label='trefoil_oblique')
plt.plot(cube_amp_rad,abé_spherique,label='abé_spherique')
plt.xlabel(f'Z{mode_mesure+1}  applied (rad RMS) ')
plt.ylabel(f'Z{mode_mesure+1} measured (rad RMS)')
plt.legend()
np.save('tilt.txt',tilt)
np.save('tip.txt',tip) 
np.save('deoc.txt',defoc)  
np.save('astig_oblique.txt',astig_oblique) 
np.save('astig_vertical.txt',astig_vertical)
np.save('trefoil_vertical.txt',trefoil_vertical)
np.save('coma_vertical.txt',coma_vertical)
np.save('coma_horizontal.txt',coma_horizontal)
np.save('trefoil_oblique.txt',trefoil_oblique)
np.save('abé_spherique.txt',abé_spherique)


#%% POUR TESTER 1 MODE courbes de responses
"""
valeurs_amp=200E-9
zwfs_amp=np.arange(-valeurs_amp, valeurs_amp,((valeurs_amp)/100))#range(1)
modes_zwfs=[(1,-1),(1,1),(2,-2),(2,0),(2,2),(3,-3),(3,-1),(3,1),(3,3),(4,0)]
step_cubephase_zwfs=[]
zernike_mode2=[]
opd=[]
vecteurs_I_R2=[]
step_cubephase=[]
sin_aberration=[]
for mode_zwfs in (modes_zwfs):
        n2, m2 = mode_zwfs
    #pour le tip tilt amplitude = amplitude_rad/(2*np.pi)
        zernike_mode2.append( (zernike_polar(theta1,rho1, n2, m2)))

for i in range (len(zwfs_amp)):
            opd.append((zwfs_amp[i]*zernike_mode2[mode_mesure]))

            E_A1_temp_zwfs=1*O*np.exp((2*np.pi*1j*opd[i])/wavelength) #sinus cycl/pupi
            #Injection of E_A in a grill with zero padding
            E_A1=np.zeros((N,N),dtype="complex128")
            ini=(N//2)-(D//2)
            end=(N//2)+(D//2)
            E_A1[ini:end,ini:end]=E_A1_temp_zwfs      
            
            
            E_B1_before_mask=MFT(N, N, N/2, E_A1,center_pixel=True)
        
            E_B1_after_mask=E_B1_before_mask*MFP 
        
            E_C1_before_LS=MFT(N, N, N/2,E_B1_after_mask,inv=True,center_pixel=True)
        
            E_C1_after_LS=(E_C1_before_LS)*D_lim_LS#*np.exp((1j*2*np.pi*opd_def)/wavelength)
        
            E_R2=MFT(N, N, N/2,E_C1_after_LS,center_pixel=True)
        
            I_R2=np.abs(E_R2)**2
            I_R2_bis=I_R2[ini1:end1,ini1:end1]
            vector_I_R2 = I_R2_bis.flatten()
            vecteurs_I_R2.append(vector_I_R2)
            #step=((control_matrix_C@((vector_I_R2-vector_I_D0))))
            #step_cubephase.append(step*(amp))
        
        
            E_R2_bis=np.zeros((NB,NB),dtype="complex128")
            ini5=(N//2)-(NB//2)
            end5=(N//2)+(NB//2)
        
            E_R2_bis=E_R2[ini5:end5,ini5:end5]
        
            E_R2_after_mask_zwfs=E_R2_bis*(MFP_zwfs)
        
            E_R2_zwfs=MFT(NB, N, m1*(N/D),E_R2_after_mask_zwfs,inv=True,center_pixel=True)
        
            E_R2_zwfs_sensor=E_C1_after_LS-((1-np.exp(1j*phase))* E_R2_zwfs)
            
            OPD_mesure_1bis=(np.abs(E_R2_zwfs_sensor)**2 )#*(wavelength/(2*np.pi))
        
        
            OPD_mesure1=OPD_mesure_1bis.flatten()#[O==1]
            step_zwfs=((control_matrix_zwfs@((OPD_mesure1-I_ref_temp))))
        
            step_cubephase_zwfs.append((step_zwfs))


tilt=[]
tip=[]
astig_oblique=[]
defoc=[]
astig_vertical=[]
trefoil_vertical=[]
coma_vertical=[]
coma_horizontal=[]
trefoil_oblique=[]
abé_spherique=[]
cycle_etude=[]
cycle_etudey=[]
cube_amp_rad=zwfs_amp
nom_modes = {
    0: ("tilt", tilt),
    1: ("tip", tip),
    2: ("astig_oblique", astig_oblique),
    3: ("defoc", defoc),
    4: ("astig_vertical", astig_vertical),
    5: ("trefoil_vertical", trefoil_vertical),
    6: ("coma_vertical", coma_vertical),
    7: ("coma_horizontal", coma_horizontal),
    8: ("trefoil_oblique", trefoil_oblique),
    9: ("abé_spherique", abé_spherique)
}
for liste in step_cubephase_zwfs:
    tilt.append(liste[0])#)*((2*np.pi)/(wavelength)))
    
    tip.append(liste[1])#)*((2*np.pi)/(wavelength)))
    astig_oblique.append(liste[2])#)*((2*np.pi)/(wavelength)))
    defoc.append(liste[3])#*((2*np.pi)/(wavelength)))
    astig_vertical.append(liste[4])#*((2*np.pi)/(wavelength)))
    
    trefoil_vertical.append(liste[5])#*((2*np.pi)/(wavelength)))
    coma_vertical.append(liste[6])#*((2*np.pi)/(wavelength)))
    coma_horizontal.append(liste[7])#*((2*np.pi)/(wavelength)))
    trefoil_oblique.append(liste[8])#*((2*np.pi)/(wavelength)))
    abé_spherique.append(liste[9])#*((2*np.pi)/(wavelength)))

    if mode_mesure in nom_modes:
        mode_name, etude_mode = nom_modes[mode_mesure]
plt.plot(zwfs_amp,zwfs_amp)       
#plt.xlim(-2E-7, 2E-7)   
#plt.ylim(-2E-7, 2E-7)   
plt.plot(zwfs_amp,etude_mode)
"""