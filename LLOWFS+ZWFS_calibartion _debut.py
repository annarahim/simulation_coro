# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 17:13:44 2024

@author: arahim
"""

import numpy as np

import matplotlib.pyplot as plt

wavelength =  1.6E-6   # Longueur d'onde de la lumière en metres
#D=8 #Diamètre de l'ouverture circulaire en m 
D = 128 # Number of pixels along the pupil diameter D
#NB = 40 # Number of pixels in the focal plane
NC=D #Nombre de pixels au plan C
NB=50
N=512   #Number of pixel of zero padding 
oversize=3 # Multiple of D for the externe diameter of RLS
D_win=30 # Zoom in a window 20x20 for plots
m1 =1.02
phase=np.pi/2

#%%Creation de l'ouverture d'entrée with zero padding 
circle_radius = (D/2) #rayon de l'ouverture en nombre de pixel
x0=((N/2))#-(1/2) #centrage du cercle au milieu de la grille de taille N
y0=((N/2))#-(1/2)

# Création d'une grille de points dans le plan XY

x1 = np.linspace(0, N-1, N)
y1 = np.linspace(0, N-1, N)
x, y = np.meshgrid(x1, y1)


#Création de l'ouverture circulaire 
aperture_radius = (x-x0)**2 + (y-y0)**2 <= circle_radius**2
#Creation obstruction central
circle_radius2=D/(2*(7.92/2.3))
rayon = (x-x0)**2 + (y-y0)**2 <= circle_radius2**2
#Creation d'une grille de 0 de taille NxN 
P=np.zeros((N,N))
P[aperture_radius]=1 #On met la valeur 1 pour l'ouverture
"""
P[rayon]=0 #On met la valeur 0 pour l'obstruction central 
epaisseur_spyder = 1  # Épaisseur de la croix
for i in range(N):
    for j in range(N):
        if aperture_radius [i, j]:
            if abs(i - j) < epaisseur_spyder or abs(i + j - (N - 1)) < epaisseur_spyder:
       
plt.figure('ouverture with zero padding')
plt.clf()
plt.title('ouverture with zero padding')
plt.imshow (P) 
plt.colorbar(label='Intensité')
"""
#%%Creation of cirle where the zernike is applied

circle_radius1 = 1 #diameter of aperture par rapport à la taille du tableau
x01=0
y01=0

# creation a point grid in the xy plane
x11 = (np.arange(D)-(D/2))/(D/2)
y11 = (np.arange(D)-(D/2))/(D/2)
x111, y111= np.meshgrid(x11, y11)
aperture_radius21 =  (x111-x01)**2 + (y111-y01)**2 <= circle_radius1**2


#Creation obstruction central 
circle_radius22=circle_radius1/(7.92/2.3) # diameter of aperture par rapport 
rayon2 = (x111-x01)**2 + (y111-y01)**2 <= circle_radius22**2
O=np.zeros((D,D))
O[aperture_radius21]=1
"""
O[rayon2]=0 

epaisseur_spyder =2  # Épaisseur de la croix en pixels 
for i in range(D):
    for j in range(D):
        if abs(i - j) < epaisseur_spyder or abs(i + j - (D - 1)) < epaisseur_spyder:
            O[i][j] = 0

"""
plt.figure()
plt.clf()
plt.title('ouverture ')
plt.imshow (O) 
plt.colorbar(label='Intensité')

#ceration spyder 
#%% ouverture pour le zwfs

aperture_radius_zwfs = (x111-x01)**2 + (y111-y01)**2 <= circle_radius1**2
O_zwfs=np.zeros((D,D))
O_zwfs[aperture_radius_zwfs]=1


#%%Creation du masque du corono 
 # Rayon du masque focal MFP
x0_MFP = (((N)/2))#-(1/2)
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

"""
plt.figure('m')
plt.clf()
plt.title('m')
plt.imshow (MFP) 
plt.colorbar(label='Intensité')
"""
#%%Creation focal mask ZWFS
circle_radius_MFP_zwfs = NB//2  #rayon du masque focal MFP
x0_MFP_zwfs=NB//2
y0_MFP_zwfs=NB//2 

# Création d'une grille de points dans le plan XY
x1_MFP_zwfs = np.linspace(0, NB-1, NB)
y1_MFP_zwfs = np.linspace(0, NB-1, NB)
x_MFP_zwfs, y_MFP_zwfs = np.meshgrid(x1_MFP_zwfs, y1_MFP_zwfs)
    
#Création de l'ouverture circulaire 

aperture_radius2_zwfs = (x_MFP_zwfs-x0_MFP_zwfs)**2 + (y_MFP_zwfs-y0_MFP_zwfs)**2 <= circle_radius_MFP_zwfs**2
MFP_zwfs=np.zeros((NB,NB))
MFP_zwfs[aperture_radius2_zwfs]=1
plt.imshow(MFP_zwfs)
"""
plt.figure('m')
plt.title('m')
plt.imshow(MFP_zwfs)
plt.colorbar(label='Intensité')
"""
#%% Creation of diameter limit for light recovery after LS 
circle_radius3 =1 #radius of aperture 
x03=0
y03=0

# creation a point grid in the xy plane
x13 = (np.arange(N)-(N/2))/(oversize*D/2)
y13 = (np.arange(N)-(N/2))/(oversize*D/2)
x3, y3 = np.meshgrid(x13, y13)
aperture_radius3 =  (x3-x03)**2 + (y3-y03)**2 <= circle_radius3**2
D_lim_LS=np.zeros((N,N))
D_lim_LS[aperture_radius3]=1
circle_radius_outer_pupill=(D/2)*0.95
aperture_radius_outer_pupill = (x-x0)**2 + (y-y0)**2 <= circle_radius_outer_pupill**2

D_lim_LS[aperture_radius_outer_pupill]=0


taille= int(1.2*circle_radius2)
# Initialisation de la grille 2D avec des espaces

# Coordonnées du coin supérieur gauche du carré dans la grille 2D
x_start = (N ) // 2
y_start = (N ) // 2
CAR_bis=np.zeros((N,N))
ini6=x_start-taille
end6=y_start+taille
"""
CAR_bis[ini6:end6,ini6:end6]=1
                     

D_lim_LS[CAR_bis==1]=1
#D_lim_LS[rayon]=1

epaisseur_D_lim = epaisseur_spyder *1.5 
for i in range(N):
    for j in range(N):
        if aperture_radius_outer_pupill[i, j]:
            if abs(i - j) < epaisseur_D_lim or abs(i + j - (N - 1)) < epaisseur_D_lim:
                D_lim_LS[i][j] = 1

plt.figure('D_lim_LS')
plt.title('D_lim_LSLS')
plt.clf()
plt.imshow(D_lim_LS)
plt.colorbar(label='Intensité')
"""
#%%

X4 = (np.arange(N)-(N/2))/(N/2)
Y4 = (np.arange(N)-(N/2))/(N/2)
x44, y44 = np.meshgrid(X4, Y4)
rho4=np.sqrt(x44**2 + y44**2) 
theta4=np.arctan2(y44,x44)
#%%

rho1 = np.sqrt(x111**2 + y111**2) 
theta1 = np.arctan2(y111,x111)
def zernike_polar(theta0,rho0, amplitude_rms, n, m):
    
    if m == 0:
        return  amplitude_rms*zernike_radial(n, m, rho0) * np.sqrt(n + 1)
    elif m > 0:
        return  amplitude_rms*zernike_radial(n, m, rho0)  * np.sqrt(2 * (n + 1))* np.cos(m * theta0)
    else:
        return  amplitude_rms*zernike_radial(n, abs(m), rho0) * np.sqrt(2 * (n + 1)) * np.sin(abs(m) * theta0)

def zernike_radial(n, m ,rho0):
    mask=rho0<=1
    rho0=rho0*mask
    result=np.zeros_like(rho0)
    for k in range((n - abs(m)) // 2 + 1):
        result += ((-1) ** k * np.math.factorial(n - k) / (np.math.factorial(k) * np.math.factorial((n + abs(m)) // 2 - k) * np.math.factorial((n - abs(m)) // 2 - k) ) * rho0 ** (n - 2 * k))
    return result


#%%Function for calculating the semi-analytical Fourier Transform 

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
#%% Determination of electric fiel without aberartions
amp_defoc=1
modes0=[(2,0,1.76E-6) ]#1.27E-6#1.8E-6
vecteurs_I0=[]

for  mode0 in (modes0):
    modes0=0
    n0, m0, amplitude_rms0 = mode0
    #pour le tip tilt amplitude = amplitude_rad/(2*np.pi)
    zernike_mode4 = zernike_polar( theta4,rho4, amplitude_rms0, n0, m0)
   

E_A0_temp=1*O#Calcul of E_A temporary in the aperture O
E_A0=np.zeros((N,N),dtype="complex128")
ini=(N//2)-(D//2)
end=(N//2)+(D//2)
E_A0[ini:end,ini:end]=E_A0_temp
#Injection of E_A in a grill with zero padding



E_B0_before_mask=MFT(N,N, N, E_A0,center_pixel=True)
E_B0_after_mask=E_B0_before_mask*MFP 

E_C0_before_LS=MFT(N, N, N,E_B0_after_mask,inv=True,center_pixel=True)
E_C0_after_LS=(E_C0_before_LS)*D_lim_LS*np.exp((1j*2*np.pi*zernike_mode4)/wavelength)


E_D0=MFT(N, N, N,E_C0_after_LS,center_pixel=True)
I_D0=np.abs(E_D0)**2


ini1=(N//2)-(D_win//2)
end1=(N//2)+(D_win//2)
I_D0_bis=I_D0[ini1:end1,ini1:end1]
 
vector_I_D0 = I_D0_bis.flatten() #colonne vecteur 1D 
vecteurs_I0.append(vector_I_D0) #liste du vecteur 1D
 
"""
plt.figure('ouverture')
plt.clf()
plt.title('ouverture')
plt.imshow(zernike_mode4)
plt.colorbar(label='Intensité')
plt.show()

plt.figure(22)
plt.clf()
plt.title('plane C  before Lyot stop ')
plt.imshow((np.abs(E_C0_before_LS)**2)**0.25)
plt.colorbar(label='Intensity')

plt.figure(33)
plt.clf()
plt.title('plane C  after Lyot stop ')
plt.imshow((np.abs(E_C0_after_LS)**2)**0.25)
plt.colorbar(label='Intensity')

plt.figure(46)
plt.clf()
plt.title('plane d ')
plt.imshow(I_D0)
plt.colorbar(label='Intensity')
"""
#%% Calibration/Interaction matrix 

amp=2.5E-9
modes=[(1,-1,amp),(1,1,amp),(2,-2,amp),(2,0,(amp)),(2,2,amp),(3,-3,amp),(3,-1,amp),(3,1,amp),(3,3,amp)]#,(4,0,amp)]
vecteurs_I_R=[]
vecteurs_I_f=[]
s=[]
b=[]
zer=[]
co=[]
vecteurs_zernike_mode=[]
Map=[]
#%%
for i, mode in enumerate(modes):
    n1, m11, amplitude_rms1 = mode
    #pour le tip tilt amplitude = amplitude_rad/(2*np.pi)
    zernike_mode = zernike_polar( theta1,rho1, amplitude_rms1, n1, m11)
    
    
    
    Map.append( zernike_polar( theta1,rho1, amplitude_rms1, n1, m11))
    vector_zernike_mode = zernike_mode.flatten()
    vecteurs_zernike_mode.append(vector_zernike_mode)
    
    E_A_temp=1*O*np.exp((1j*2*np.pi*zernike_mode)/wavelength) # electric field  plane A 
    E_A=np.zeros((N,N),dtype="complex128")
    ini=(N//2)-(D//2)
    end=(N//2)+(D//2)
    E_A[ini:end,ini:end]=E_A_temp
 
 
    #Calcul of the direct TF of the electric field at plane B 
    E_B_before_mask=MFT(N, N, N, E_A,center_pixel=True)
    E_B_after_mask=E_B_before_mask*MFP  #Multiplying the B electric field by the mask 
    
       
    # PLANE C
    #Calcul of the inverse TF of the elec field in plane B  
    E_C_before_LS= MFT(N, N, N,E_B_after_mask,inv=True,center_pixel=True)
    #Determination of the elec field at the C plane multiplied by the lyot stop 

    E_C_after_LS=(E_C_before_LS)*D_lim_LS*np.exp((1j*2*np.pi*zernike_mode4)/wavelength)
 
   
    # PLANE D 
    #Calcul of the TF of the elec field in plane C giving the elec field in plane D 
    E_R = MFT(N, N, N,E_C_after_LS,center_pixel=True)
    I_R=np.abs(E_R)**2
    
    ini1=(N//2)-(D_win//2)
    end1=(N//2)+(D_win//2)
    I_Rbis=I_R[ini1:end1,ini1:end1]
    vector_I_R = I_Rbis.flatten()
    vecteurs_I_R.append(vector_I_R)
    
    #Determination des abérattions
    #I_f=(I_R-I_D0)/amplitude_rms
    I_f=(vector_I_R-vector_I_D0)#/amplitude_rms
    vecteurs_I_f.append(I_f)
    

vecteur = np.array(vecteurs_I_f)
s=vecteur.T
#tableau_2D = vecteur_1D.reshape(N, N)
control_matrix_C = np.linalg.pinv(s)
"""
plt.figure(46)
plt.clf()
plt.title('plane d ')
plt.imshow(vecteur)
plt.colorbar(label='Intensity')
"""




#%%Calibration/Interaction matrix 
valeur_amp_calib=20E-9
amp_calib=[valeur_amp_calib, -valeur_amp_calib]
zwfs_modes=[(1,-1,amp_calib),(1,1,amp_calib),(2,-2,amp_calib),(2,0,amp_calib),(2,2,amp_calib),(3,-3,amp_calib),(3,-1,amp_calib),(3,1,amp_calib),(3,3,amp_calib)]#,(4,0,amp_calib)]
vecteurs_I_R_zwfs=[]
vecteurs_I_f_zwfs=[]
vecteurs_mean=[]
tab=[]
s=[]
zernike_mode_zwfs=[]
I=[]
images=[]
#%%

for i, ce in enumerate(amp_calib):
    for u, zwfs_mode in enumerate (zwfs_modes):
        n1, m11, amplitude_rms1 =zwfs_mode
        
            
            # Utiliser une valeur différente de cube_amp
        amplitude_rms1 = amp_calib[i] 
        #pour le tip tilt amplitude = amplitude_rad/(2*np.pi)
        zernike_mode_zwfs.append(zernike_polar(theta1,rho1, amplitude_rms1, n1, m11))
     
#%%

for q, zern in enumerate (zernike_mode_zwfs):
    
    E_A_temp_zwfs=1*O*np.exp((1j*2*np.pi*zernike_mode_zwfs[q])/wavelength) # electric field  plane A 
    E_A_zwfs=np.zeros((N,N),dtype="complex128")
    ini=(N//2)-(D//2)
    end=(N//2)+(D//2)
    E_A_zwfs[ini:end,ini:end]=E_A_temp_zwfs
    
    
    E_before_mask=MFT(N, N, N, E_A_zwfs,center_pixel=True)
    E_after_mask=E_before_mask*MFP 

    E_before_LS=MFT(N, N, N,E_after_mask,inv=True,center_pixel=True)
    E_after_LS=(E_before_LS)*D_lim_LS*np.exp((1j*2*np.pi*zernike_mode4)/wavelength)
    E=MFT(N, N, N,E_after_LS,center_pixel=True)
    I_E=np.abs(E)**2
    I_E_bis=I_E[ini1:end1,ini1:end1]
    I_R2_zzz=I_E[ini:end,ini:end]
    E_zw=MFT(N, N, N,E,inv=True,center_pixel=False)
     
 
    #Calcul of the direct TF of the electric field at plane B 
    E_B_before_mask_zwfs=MFT(N, NB, m1*(N/D), E_zw,center_pixel=False)
    E_B_after_mask_zwfs=E_B_before_mask_zwfs*(MFP_zwfs)   #Multiplying the B electric field by the mask 
    E_C0_zwfs= MFT(NB, N, m1*(N/D),E_B_after_mask_zwfs,inv=True,center_pixel=False)
       
    # PLANE C
    #Calcul of the inverse TF of the elec field in plane B  
    E_C_zwfs= E_zw-((1-np.exp(1j*phase))* E_C0_zwfs)
    #Determination of the elec field at the C plane multiplied by the lyot stop 

 
   
    # PLANE D 
    #Calcul of the TF of the elec field in plane C giving the elec field in plane D 
    
    I_R_zwfs=np.abs(E_C_zwfs)**2
    I.append(I_R_zwfs)
    mean_I_R=np.mean(I_R_zwfs)
    
    vector_I_R_zwfs = I_R_zwfs.flatten()
    
    
    I_f_zwfs=(vector_I_R_zwfs)#-vector_I_5)
    
    tab.append(I_f_zwfs)
    vecteurs_I_R_zwfs.append(vector_I_R_zwfs)
    vecteurs_mean.append(mean_I_R)
    image=np.reshape(I_f_zwfs,(512,512))
    images.append(image)
    #Determination des abérattions
    #I_f=(I_R-I_D0)/amplitude_rms
    #/amplitude_rms
    #vecteurs_I_f.append(I_f)
    
    """
    plt.figure(q*20)
    plt.clf()
    plt.title('abé')
    #plt.imshow(np.abs(E_B_after_mask_zwfs)**2)
    plt.imshow(image,vmin=-0.5,vmax=0.5)
    plt.colorbar(label='Intensity')
    """


n_images = len(images)

# Déterminer la taille de la grille de subplots
n_cols = 5  # Nombre de colonnes
n_rows = (n_images + n_cols - 1) // n_cols  # Calculer le nombre de lignes nécessaires

# Créer une figure et une grille de subplots
fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_images, 3 * n_rows))

# Boucle sur chaque image
for i, image in enumerate(images):
    row = i // n_cols
    col = i % n_cols
    
    ax = axs[row, col] if n_rows > 1 else axs[col]  
    if i < len(images)/2 :
        ax.set_title(f'Z {i+1} - abé {valeur_amp_calib}  ')
    else : 
        ax.set_title(f'Z {i+1-(len(images)/2)} - abé {-valeur_amp_calib}  ')
    im = ax.imshow(image)#, vmin=-0.5, vmax=0.5)
    fig.colorbar(im, ax=ax, label='Intensity')

# Si le nombre d'images est inférieur au nombre de subplots, masquer les axes inutilisés
if n_images < n_rows * n_cols:
    for j in range(n_images, n_rows * n_cols):
        fig.delaxes(axs.flatten()[j])

plt.tight_layout()
plt.show()

sum=[]
taille=int(len(tab)/2)
for indice in range(taille):
    
    sum.append(((tab[indice])-tab[indice+len(zwfs_modes)])/((2*amp_calib[0])))
    

vecteur_zwfs = np.array(sum)
s_zwfs=vecteur_zwfs.T
#tableau_2D = vecteur_1D.reshape(N, N)
control_matrix_zwfs = np.linalg.pinv(s_zwfs) 
#%%
plt.imshow(images[0]-images[8])
plt.colorbar(label='Intensity')
"""
abci= ((np.arange(N)))

#plt.colorbar(label='Intensity')
plt.plot(abci,np.log10(images[0][N//2,0:N]),label='z1')
plt.plot(abci,np.log10(images[1][N//2,0:N]),label='z2')
plt.plot(abci,np.log10(images[2][N//2,0:N]),label='z3')
plt.plot(abci,np.log10(images[3][N//2,0:N]),label='z4')
plt.plot(abci,np.log10(images[4][N//2,0:N]),label='z5')
plt.plot(abci,np.log10(images[5][N//2,0:N]),label='z6')
plt.plot(abci,np.log10(images[6][N//2,0:N]),label='z7')
plt.plot(abci,np.log10(images[7][N//2,0:N]),label='z8')
plt.plot(abci,np.log10(images[8][N//2,0:N]),label='z9')
plt.legend()
"""
#%% for create sensor response
modes2=[(1,-1,20E-9),(1,1,0E-9),(2,-2,0E-9),(2,0,0E-9),(2,2,0E-9),(3,-3,0),(3,-1,0E-9),(3,1,0),(3,3,0)]#,(4,0,0E-9)]
vecteurs_I_R2=[]
zernike_mode2=[]
step_cubephase=[]
for  mode2 in (modes2):
    n2, m2, amplitude_rms2 = mode2
    #pour le tip tilt amplitude = amplitude_rad/(2*np.pi)
    zernike_mode2.append(zernike_polar(theta1,rho1, amplitude_rms2, n2, m2))
    
        
zernike_mode22= zernike_mode2[0]+  zernike_mode2[1] + zernike_mode2[2]  + zernike_mode2[3]   + zernike_mode2[4]  + zernike_mode2[5]    + zernike_mode2[6]  + zernike_mode2[7]    + zernike_mode2[8] # + zernike_mode2[9]    

"""
plt.figure('tests')
plt.clf()
plt.title('zernike_mode22')
plt.imshow(zernike_mode2[0])
plt.colorbar(label='Intensity')
"""

#%% ZWFS
#creation d'une image sans aberation pour la calibration cad avoir
#la lumière au plan apres le LLOWFS à calibrer et pas en entrée 

E_1_temp=1*O#*np.exp((1j*2*np.pi*zernike_mode22)/wavelength) #Calcul of E_A temporary in the aperture O

#Injection of E_A in a grill with zero padding
E_1=np.zeros((N,N),dtype="complex128")
ini=(N//2)-(D//2)
end=(N//2)+(D//2)
E_1[ini:end,ini:end]=E_1_temp


E_2_before_mask=MFT(N, N, N, E_1,center_pixel=True)
E_2_after_mask=E_2_before_mask*MFP 

E_3_before_LS=MFT(N, N, N,E_2_after_mask,inv=True,center_pixel=True)
E_3_after_LS=(E_3_before_LS)*D_lim_LS*np.exp((1j*2*np.pi*zernike_mode4)/wavelength)
E_4=MFT(N, NB, NB,E_3_after_LS,center_pixel=True)

E_5_0=E_4*MFP_zwfs
E_5=MFT(NB, N,NB ,E_5_0,inv=True,center_pixel=True)
E_cam= E_3_after_LS-((1-np.exp(1j*phase))* E_5)
#test=MFT(N, N,N ,E_cam,center_pixel=True)

I_5=np.abs(E_5)**2
I_5_bis=I_5[ini:end,ini:end]

vector_I_5 = I_5.flatten()
plt.imshow((I_5))
plt.title("E_cam image au plan du zwfs")
plt.colorbar(label='Intensity')

np.save('A1.txt',I_5)
#%%
a=np.load(('A.txt.npy'))
b=np.load(('A1.txt.npy'))

plt.imshow(a-b)
plt.title("(3)")
plt.colorbar(label='Intensity')
#%%LLOWFS
E_A1_temp=1*O*np.exp((1j*2*np.pi*zernike_mode22)/wavelength) #Calcul of E_A temporary in the aperture O

#Injection of E_A in a grill with zero padding
E_A1=np.zeros((N,N),dtype="complex128")
ini=(N//2)-(D//2)
end=(N//2)+(D//2)
E_A1[ini:end,ini:end]=E_A1_temp


E_B1_before_mask=MFT(N, N, N, E_A1,center_pixel=True)
E_B1_after_mask=E_B1_before_mask*MFP 

E_C1_before_LS=MFT(N, N, N,E_B1_after_mask,inv=True,center_pixel=True)
E_C1_after_LS=(E_C1_before_LS)*D_lim_LS*np.exp((1j*2*np.pi*zernike_mode4)/wavelength)
E_R2=MFT(N, N, N,E_C1_after_LS,center_pixel=True)
I_R2=np.abs(E_R2)**2
I_R2_bis=I_R2[ini1:end1,ini1:end1]
I_R2_z=I_R2[ini:end,ini:end]
E_a=MFT(N, N, N,E_R2,inv=True,center_pixel=False)
I_aa=np.abs(E_a)**2
"""
E_a_zwfs=np.zeros((N,N),dtype="complex128")
ini=(N//2)-(D//2)
end=(N//2)+(D//2)
E_a_zwfs[ini:end,ini:end]=E_a
I_a=np.abs(E_a_zwfs)**2
#plt.imshow(I_a)
"""
vector_I_R2 = I_R2_bis.flatten()
vecteurs_I_R2.append(vector_I_R2)
step=((control_matrix_C@((vector_I_R2-vector_I_D0))))
step_cubephase.append(step*(amp))

#%%ZWFS

"""

E_A1_temp_zwfs=1*E_a_zwfs#*np.exp((1j*2*np.pi*zernike_mode_zwfs)/wavelength) #Calcul of E_A temporary in the aperture O

#Injection of E_A in a grill with zero padding
E_A1_zwfs=np.zeros((N,N),dtype="complex128")
ini=(N//2)-(D//2)
end=(N//2)+(D//2)
E_A1_zwfs[ini:end,ini:end]=E_A1_temp_zwfs

"""
E_B1_before_mask_zwfs=MFT(N, NB, m1*(N/D), E_a,center_pixel=False)
E_B1_after_mask_zwfs=E_B1_before_mask_zwfs*(MFP_zwfs)
E_C10_zwfs=MFT(NB, N, m1*(N/D),E_B1_after_mask_zwfs,inv=True,center_pixel=False)


E_C1_zwfs=E_a-((1-np.exp(1j*phase))* E_C10_zwfs)
#plt.imshow(np.abs(E_B1_after_mask_zwfs)**2)
#plt.colorbar(label='Intensity')
I_R2_zwfs=np.abs(E_C1_zwfs)**2 
#plt.imshow(I_aa)
#plt.imshow(I_R2_zwfs-I_aa)
vector_I_R2_zwfs = I_R2_zwfs.flatten()
step_zwfs=((control_matrix_zwfs@((vector_I_R2_zwfs))))
