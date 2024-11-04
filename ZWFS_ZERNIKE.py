# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 17:39:26 2024

@author: arahim
"""

import numpy as np
import matplotlib.pyplot as plt

#%% VARIABLE DEFINTIONS
wavelength = 1.625E-6 # wavelenght in meters
D=750# Diameter of circular aperture in meters
N=D #Number of pixel for zerro padding 
NB=50# Number of pixels in the focal plane
m1 =1.06#sampling (wavelenght/D)
phase=np.pi/2

mode_mesure=9


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
        circle_radius2 = D / (2 * (7.92 / 2.3))
        central_obstruction = (x - x0) ** 2 + (y - y0) ** 2 <= circle_radius2 ** 2
        P[central_obstruction] = 0

        # Création des croisillons (spyder)
        for i in range(N):
            for j in range(N):
                if aperture_radius[i, j]:
                    if abs(i - j) < epaisseur_spyder or abs(i + j - (N - 1)) < epaisseur_spyder:
                        P[i, j] = 0

    return P


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

plt.figure('m')
plt.title('m')
plt.imshow(MFP_zwfs)
plt.colorbar(label='Intensité')



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
#%%Choix aperture
aperture='circular'
P = create_aperture(N, D, type=aperture)
start = (N - D) // 2
O = P[start:start + D, start:start + D]
#%%creation d'une boucle qui creer un sinus de cycl allant de 0 à 400 
cycles_max =int( D/2)
abe_fourier_x=[]
abe_fourier_y=[]
for cycles in range(0, cycles_max , 1):  # Incrémente de 50 cycles à chaque étape
    abe_fourier_x.append( np.sin(np.pi*cycles*x111))
    abe_fourier_y.append( np.sin(np.pi*cycles*y111))
    
#%% Calibration/Interaction matrix 

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
    
    E_A_zwfs_temp=1*O*np.exp((1j*2*np.pi*OPD_calib[q])/wavelength) # electric field  plane A 
    E_A_zwfs=np.zeros((N,N),dtype="complex128")
    ini=(N//2)-(D//2)
    end=(N//2)+(D//2)
    E_A_zwfs[ini:end,ini:end]=E_A_zwfs_temp
    

    

    #Calcul of the direct TF of the electric field at plane B 
    E_B_before_mask_zwfs=MFT(N, NB, m1*(N/D), E_A_zwfs,center_pixel=True)
    E_B_after_mask_zwfs=E_B_before_mask_zwfs*(MFP_zwfs)   #Multiplying the B electric field by the mask 
    E_C0_zwfs= MFT(NB, N, m1*(N/D),E_B_after_mask_zwfs,inv=True,center_pixel=True)
       
    # PLANE C
    #Calcul of the inverse TF of the elec field in plane B  
    E_C_zwfs= E_A_zwfs-(1-np.exp(1j*phase))* E_C0_zwfs
    #Determination of the elec field at the C plane multiplied by the lyot stop 

 
   
    # PLANE D 
    #Calcul of the TF of the elec field in plane C giving the elec field in plane D 
    
    OPD_mesure=(np.abs(E_C_zwfs)**2)#*(wavelength/(2*np.pi))
       
        
    image=OPD_mesure
    
    OPD_mesure_bis=OPD_mesure.flatten()#[O==1]
    tab.append(OPD_mesure_bis)
    images.append(OPD_mesure)

#%%
sum1=[]
sum2=[]
images_sum=[]
images_sum2=[]
taille=int(len(tab)/2)
indice_de_depart=2*taille

for indice in range(taille):
    
    sum1.append(((tab[indice])-tab[indice+len(zwfs_modes)])/((2*amp_calib[0])))
    images_sum.append(((images[indice])-images[indice+len(zwfs_modes)])/((2*amp_calib[0])))



#%%
vecteur_zwfs = np.array(sum1)#sum1 pour zernike
s_zwfs=vecteur_zwfs.T
control_matrix_zwfs = np.linalg.pinv(s_zwfs) 
#%% affichage   pour zernike 
""" 
n_images = len(images)

# Déterminer la taille de la grille de subplots
n_cols = 5   # Nombre de colonnes
n_rows = (n_images + n_cols - 1) // n_cols  # Calculer le nombre de lignes nécessaires

# Créer une figure et une grille de subplots
fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_images, 3 * n_rows))


# Boucle sur chaque image
for i, image in enumerate(images):
    row = i // n_cols
    col = i % n_cols
    
    ax = axs[row, col] if n_rows > 1 else axs[col]  
    
    im = ax.imshow(images[i])#, vmin=-0.5, vmax=0.5)
    fig.colorbar(im, ax=ax, label='Intensity')

# Si le nombre d'images est inférieur au nombre de subplots, masquer les axes inutilisés
if n_images < n_rows * n_cols:
    for j in range(n_images, n_rows * n_cols):
        fig.delaxes(axs.flatten()[j])

plt.tight_layout()
plt.show()   
""" 

#%% affichage pour zernike  
""" 
n_images = len(images_sum)

# Déterminer la taille de la grille de subplots
n_cols = 5   # Nombre de colonnes
n_rows = (n_images + n_cols - 1) // n_cols  # Calculer le nombre de lignes nécessaires

# Créer une figure et une grille de subplots
fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_images, 3 * n_rows))


# Boucle sur chaque image
for i, image in enumerate(images_sum):
    row = i // n_cols
    col = i % n_cols
    
    ax = axs[row, col] if n_rows > 1 else axs[col]  
    
    im = ax.imshow(images_sum[i])#, vmin=-0.5, vmax=0.5)
    fig.colorbar(im, ax=ax, label='Intensity')

# Si le nombre d'images est inférieur au nombre de subplots, masquer les axes inutilisés
if n_images < n_rows * n_cols:
    for j in range(n_images, n_rows * n_cols):
        fig.delaxes(axs.flatten()[j])

plt.tight_layout()
plt.show() 
"""   
#%%ref

E_temp=1*O
E=np.zeros((N,N),dtype="complex128")
ini=(N//2)-(D//2)
end=(N//2)+(D//2)
E[ini:end,ini:end]=E_temp

E_ref_before_mask_zwfs=MFT(N, NB, m1*(N/D), E,center_pixel=True)
E_ref_after_mask_zwfs=E_ref_before_mask_zwfs*(MFP_zwfs)
E_C_ref=MFT(NB, N, m1*(N/D),E_ref_after_mask_zwfs,inv=True,center_pixel=True)
E_ref_zwfs=E-((1-np.exp(1j*phase))* E_C_ref)
I_ref_temp=np.abs(E_ref_zwfs)**2
I_ref=I_ref_temp.flatten()#[O==1]


#%%POUR TESTER TOUS LES MODES EN MEME TEMPS
valeurs_amp=200E-9
zwfs_amp=np.arange(-valeurs_amp, valeurs_amp,((valeurs_amp)/100))#range(1)
modes_zwfs=[(1,-1),(1,1),(2,-2),(2,0),(2,2),(3,-3),(3,-1),(3,1),(3,3),(4,0)]
step_cubephase_zwfs=[]
zernike_mode2=[]
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
        
        


        #E_A1_temp_zwfs=1*O*np.exp((1j*2*np.pi*opd[i])/wavelength) 
       
    
        #Injection of E_A in a grill with zero padding
        E_A1_zwfs=np.zeros((N,N),dtype="complex128")
        ini=(N//2)-(D//2)
        end=(N//2)+(D//2)
        E_A1_zwfs[ini:end,ini:end]=E_A1_temp_zwfs
    
    
        E_B1_before_mask_zwfs=MFT(N, NB, m1*(N/D), E_A1_zwfs,center_pixel=True)
        E_B1_after_mask_zwfs=E_B1_before_mask_zwfs*(MFP_zwfs)
        E_C10_zwfs=MFT(NB, N, m1*(N/D),E_B1_after_mask_zwfs,inv=True,center_pixel=True)
    
        E_C1_zwfs=E_A1_zwfs-((1-np.exp(1j*phase))* E_C10_zwfs)
    
        OPD_mesure_1bis=(np.abs(E_C1_zwfs)**2 )#*(wavelength/(2*np.pi))
       
       
        
        OPD_mesure1=OPD_mesure_1bis.flatten()#[O==1]
       
        step_zwfs=((control_matrix_zwfs@((OPD_mesure1-I_ref))))
        step_cubephase_zwfs.append(step_zwfs)
        
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
#plt.plot(cube_amp_rad,abé_spherique)
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

#%%POUR TESTER 1 MODE
"""
valeurs_amp=200E-9
zwfs_amp=np.arange(-valeurs_amp, valeurs_amp,((valeurs_amp)/100))#range(1)
modes_zwfs=[(1,-1),(1,1),(2,-2),(2,0),(2,2),(3,-3),(3,-1),(3,1),(3,3),(4,0)]
step_cubephase_zwfs=[]
zernike_mode2=[]
opd=[]

for mode_zwfs in (modes_zwfs):
        n2, m2 = mode_zwfs
        zernike_mode2.append( (zernike_polar(theta1,rho1, n2, m2)))
for i in range (len(zwfs_amp)):
        opd.append((zwfs_amp[i]*zernike_mode2[mode_mesure]))


        E_A1_temp_zwfs=1*O*np.exp((1j*2*np.pi*opd[i])/wavelength) 
       
    
        #Injection of E_A in a grill with zero padding
        E_A1_zwfs=np.zeros((N,N),dtype="complex128")
        ini=(N//2)-(D//2)
        end=(N//2)+(D//2)
        E_A1_zwfs[ini:end,ini:end]=E_A1_temp_zwfs
    
    
        E_B1_before_mask_zwfs=MFT(N, NB, m1*(N/D), E_A1_zwfs,center_pixel=True)
        E_B1_after_mask_zwfs=E_B1_before_mask_zwfs*(MFP_zwfs)
        E_C10_zwfs=MFT(NB, N, m1*(N/D),E_B1_after_mask_zwfs,inv=True,center_pixel=True)
    
        E_C1_zwfs=E_A1_zwfs-((1-np.exp(1j*phase))* E_C10_zwfs)
    
        OPD_mesure_1bis=(np.abs(E_C1_zwfs)**2 )#*(wavelength/(2*np.pi))
       
       
        
        OPD_mesure1=OPD_mesure_1bis.flatten()#[O==1]
       
        step_zwfs=((control_matrix_zwfs@((OPD_mesure1-I_ref))))
        step_cubephase_zwfs.append(step_zwfs)
        

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
#np.save(f'test_zwfs{mode_mesure}.txt',etude_mode) 
"""
#%%
"""
plt.figure('response')
plt.plot(zwfs_amp,etude_mode)
plt.title(f"{mode_name}")


plt.figure('linear response')
plt.plot(zwfs_amp,zwfs_amp)
plt.plot(zwfs_amp,tilt,label=f'mode mesuré z{mode_mesure +1}')
plt.plot(cube_amp_rad,tip,label='tip')
plt.plot(cube_amp_rad,tilt,"--",label='tilt')
plt.plot(cube_amp_rad,defoc,"--",label='defocus')
plt.plot(cube_amp_rad,astig_oblique,"--",label='astig_oblique')
plt.plot(cube_amp_rad,astig_vertical,"--",label='astig_vertical')
plt.plot(cube_amp_rad, trefoil_vertical,"--",label=' trefoil_vertical')
plt.plot(cube_amp_rad,coma_vertical,"--",label='coma_vertical')
plt.plot(cube_amp_rad,coma_horizontal,"--",label='coma_horizontal')
plt.plot(cube_amp_rad,trefoil_oblique,"--",label='trefoil_oblique')
plt.plot(cube_amp_rad,abé_spherique,"--",label='abé_spherique')
plt.legend()
"""
#%%
"""
cube_amp_rad=zwfs_amp

a1=np.load('test_zwfs0.txt.npy')
a=np.load('test_zwfs1.txt.npy')
b=np.load('test_zwfs2.txt.npy')
c=np.load('test_zwfs3.txt.npy')
d=np.load('test_zwfs4.txt.npy')
e=np.load('test_zwfs5.txt.npy')
f=np.load('test_zwfs6.txt.npy')
g=np.load('test_zwfs7.txt.npy')
h=np.load('test_zwfs8.txt.npy')
i=np.load('test_zwfs9.txt.npy')

plt.plot(cube_amp_rad,a1,"--",label='tilt')
plt.plot(cube_amp_rad,a,"--",label='tip')
plt.plot(cube_amp_rad,b,"--",label='astig oblique')
plt.plot(cube_amp_rad,c,"--",label='defocus')
plt.plot(cube_amp_rad,d,"--",label='astig vertical')
plt.plot(cube_amp_rad,e,"--",label='trefoil vertical')
plt.plot(cube_amp_rad,f,"--",label='coma vertical')
plt.plot(cube_amp_rad,g,"--",label='coma horizontal')
plt.plot(cube_amp_rad,h,"--",label='trefoil oblique ')
plt.plot(cube_amp_rad,i,"--",label='abe spherique ')
plt.plot(cube_amp_rad,cube_amp_rad,"-",label='reponse unitaire ')

plt.xlabel('error applied (nm)')
plt.ylabel('error measured (nm)')
plt.title('ZWFS CIRCULAR mask = 1.06 \u03BB /D')
plt.legend()
"""