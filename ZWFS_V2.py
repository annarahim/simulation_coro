# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 17:45:55 2024

@author: arahim
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#%% VARIABLE DEFINTIONS
wavelength = 1.6E-6 # wavelenght in meters
D_tel=8 #Diameter of telescopes in meters
D=128# Diameter of circular aperture in meters
N=D #Number of pixel for zerro padding 
NA = D# Number of pixels along the pupil diameter D
NB=50# Number of pixels in the focal plane
NC=NA #Number of pixels in plane C
ND=75#Number of pixels in plane D
m1 =1.06#sampling (wavelenght/D)
md =20# 20 sampling for TF in plane D 
D_win=200
phase=np.pi/2

#%%Creation of circular aperture
#%%Creation de l'ouverture d'entrée with zero padding 

circle_radius = (D/2) #rayon de l'ouverture en nombre de pixel
x0=((N/2))#-(1/2) #centrage du cercle au milieu de la grille de taille N
y0=((N/2))#-(1/2)

# Création d'une grille de points dans le plan XY

x1 = np.linspace(0+0.5, N-1+0.5, N)
y1 = np.linspace(0+0.5, N-1+0.5, N)
x, y = np.meshgrid(x1, y1)


#Création de l'ouverture circulaire 
aperture_radius = (x-x0)**2 + (y-y0)**2 <= circle_radius**2

#Creation d'une grille de 0 de taille NxN 
P=np.zeros((N,N))
P[aperture_radius]=1 #On met la valeur 1 pour l'ouverture
#plt.imshow(P-np.fliplr(P))
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
#%%Creation of cirle where the zernike is applied

circle_radius1 = 1 #diameter of aperture par rapport à la taille du tableau
x01=0
y01=0

# creation a point grid in the xy plane
x11 = (np.arange(D)-(D/2)+0.5)/(D/2)
y11 = (np.arange(D)-(D/2)+0.5)/(D/2)
x111, y111= np.meshgrid(x11, y11)
aperture_radius21 =  (x111-x01)**2 + (y111-y01)**2 <= circle_radius1**2

O=np.zeros((D,D))
O[aperture_radius21]=1
#plt.imshow(O-np.flipud(O))
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


#%% Calibration/Interaction matrix 

amp_calib=[20E-9, -20E-9]
zwfs_modes=[(1,-1,amp_calib),(1,1,amp_calib),(2,-2,amp_calib),(2,0,amp_calib),(2,2,amp_calib),(3,-3,amp_calib),(3,-1,amp_calib),(3,1,amp_calib),(3,3,amp_calib),(4,0,amp_calib)]
vecteurs_I_R_zwfs=[]
vecteurs_I_f_zwfs=[]
vecteurs_mean=[]
tab=[]
s=[]
zernike_mode_zwfs=[]
I=[]

#%%

for i, ce in enumerate(amp_calib):
    for u, zwfs_mode in enumerate (zwfs_modes):
        n1, m11, amplitude_rms1 =zwfs_mode
        
            
            # Utiliser une valeur différente de cube_amp
        amplitude_rms1 = amp_calib[i] 
        #pour le tip tilt amplitude = amplitude_rad/(2*np.pi)
        zernike_mode_zwfs.append(zernike_polar(theta1,rho1, amplitude_rms1, n1, m11))
for q, zern in enumerate (zernike_mode_zwfs):
    
    E_A_zwfs=1*O*np.exp((1j*2*np.pi*zernike_mode_zwfs[q])/wavelength) # electric field  plane A 

    
 
 
    #Calcul of the direct TF of the electric field at plane B 
    E_B_before_mask_zwfs=MFT(D, NB, m1*(N/D), E_A_zwfs,center_pixel=True)
    E_B_after_mask_zwfs=E_B_before_mask_zwfs*(MFP_zwfs)   #Multiplying the B electric field by the mask 
    E_C0_zwfs= MFT(NB, D, m1*(N/D),E_B_after_mask_zwfs,inv=True,center_pixel=True)
       
    # PLANE C
    #Calcul of the inverse TF of the elec field in plane B  
    E_C_zwfs= E_A_zwfs-((1-np.exp(1j*phase))* E_C0_zwfs)
    #Determination of the elec field at the C plane multiplied by the lyot stop 

 
   
    # PLANE D 
    #Calcul of the TF of the elec field in plane C giving the elec field in plane D 
    
    I_R_zwfs=np.abs(E_C0_zwfs)**2
    I.append(I_R_zwfs)
    mean_I_R=np.mean(I_R_zwfs)
    
    vector_I_R_zwfs = I_R_zwfs.flatten()
    tab.append(vector_I_R_zwfs)
    vecteurs_I_R_zwfs.append(vector_I_R_zwfs)
    vecteurs_mean.append(mean_I_R)
     
    #Determination des abérattions
    #I_f=(I_R-I_D0)/amplitude_rms
    I_f_zwfs=(vector_I_R_zwfs/mean_I_R)#/amplitude_rms
    #vecteurs_I_f.append(I_f)
    
    """
    plt.figure(q*20)
    plt.clf()
    plt.title('I_R')
    plt.imshow(I_R_zwfs)
    plt.colorbar(label='Intensity')
    """
    
    
    
    
    
sum=[]
taille=int(len(tab)/2)
for indice in range(taille):
    
    sum.append(((tab[indice])-tab[indice+len(zwfs_modes)])/((2*amp_calib[0])))
    

vecteur_zwfs = np.array(sum)
s_zwfs=vecteur_zwfs.T
#tableau_2D = vecteur_1D.reshape(N, N)
control_matrix_zwfs = np.linalg.pinv(s_zwfs) 

#%% for create sensor response
modes2=[(1,-1,0E-9),(1,1,0E-9),(2,-2,0E-9),(2,0,30E-9),(2,2,0E-9),(3,-3,0),(3,-1,0E-9),(3,1,0),(3,3,0),(4,0,0E-9)]
vecteurs_I_R2=[]
zernike_mode2=[]
step_cubephase_zwfs=[]
for  mode2 in (modes2):
    n2, m2, amplitude_rms2 = mode2
    #pour le tip tilt amplitude = amplitude_rad/(2*np.pi)
    zernike_mode2.append(zernike_polar(theta1,rho1, amplitude_rms2, n2, m2))
    
        
zernike_mode22= zernike_mode2[0]+  zernike_mode2[1] + zernike_mode2[2]  + zernike_mode2[3]   + zernike_mode2[4]  + zernike_mode2[5]    + zernike_mode2[6]  + zernike_mode2[7]    + zernike_mode2[8]  + zernike_mode2[9]    

"""
plt.figure('tests')
plt.clf()
plt.title('zernike_mode22')
plt.imshow(zernike_mode22)#-np.flipud(zernike_mode22))
plt.colorbar(label='Intensity')
"""

#%%
"""

E_A1_temp_zwfs=1*O*np.exp((1j*2*np.pi*zernike_mode22)/wavelength)#*np.exp((1j*2*np.pi*zernike_mode_zwfs)/wavelength) #Calcul of E_A temporary in the aperture O

#Injection of E_A in a grill with zero padding

E_A1_zwfs=np.zeros((N,N),dtype="complex128")
ini=(N//2)-(D//2)
end=(N//2)+(D//2)
E_A1_zwfs[ini:end,ini:end]=E_A1_temp_zwfs


E_B1_before_mask_zwfs=MFT(D, NB, m1*(D/D), E_A1_temp_zwfs,center_pixel=False)
E_B1_after_mask_zwfs=E_B1_before_mask_zwfs*(MFP_zwfs)
E_C10_zwfs=MFT(NB, D, m1*(D/D),E_B1_after_mask_zwfs,inv=True,center_pixel=False)

E_C1_zwfs=E_A1_temp_zwfs-((1-np.exp(1j*phase))* E_C10_zwfs)

I_R2_zwfs=np.abs(E_C1_zwfs)**2 

plt.figure('tests2')
plt.clf()
plt.title('zernike_mode222')
plt.imshow(np.abs((E_C1_zwfs)**2))
plt.colorbar(label='Intensity')

vector_I_R2_zwfs = I_R2_zwfs.flatten()
step_zwfs=((control_matrix_zwfs@((vector_I_R2_zwfs))))

step_cubephase_zwfs.append(step_zwfs)
    
"""

#%%

valeurs_amp=40E-9
zwfs_amp=np.arange(-valeurs_amp, valeurs_amp,((valeurs_amp)/100))#range(1)
modes_zwfs=[(1,-1,0),(1,1,zwfs_amp),(2,-2,0),(2,0,0),(2,2,0),(3,-3,0),(3,-1,0),(3,1,0),(3,3,0),(4,0,0)]
vecteurs_I_R_zwfs=[]
zernike_mode_zwfs=[]
step_cubephase_zwfs=[]
sommes_zwfs=[]
for i, zwfs in enumerate(zwfs_amp):
    for u, mode_zwfs in enumerate (modes_zwfs):
        n2, m2, amplitude_rms2 = mode_zwfs
        if n2 ==1 and m2 ==1:
            
            # Utiliser une valeur différente de cube_amp
            amplitude_rms2 = zwfs_amp[i] 
        #pour le tip tilt amplitude = amplitude_rad/(2*np.pi)
        zernike_mode_zwfs.append(zernike_polar(theta1,rho1, amplitude_rms2, n2, m2))
    #zernike_mode22=np.array(zernike_mode2)
for t in range(0, len(zernike_mode_zwfs), len(modes_zwfs)):
    groupe_tableaux = zernike_mode_zwfs[t:t+(len(modes_zwfs))]
    Z_zwfs=np.sum(groupe_tableaux, axis=0)
    sommes_zwfs.append(Z_zwfs)
for i, cube in enumerate(zwfs_amp):
    E_A1_temp_zwfs=1*O*np.exp((1j*2*np.pi*sommes_zwfs[i])/wavelength) #Calcul of E_A temporary in the aperture O

    #Injection of E_A in a grill with zero padding
    E_A1_zwfs=np.zeros((N,N),dtype="complex128")
    ini=(N//2)-(D//2)
    end=(N//2)+(D//2)
    E_A1_zwfs[ini:end,ini:end]=E_A1_temp_zwfs


    E_B1_before_mask_zwfs=MFT(N, NB, m1*(N/D), E_A1_zwfs,center_pixel=True)
    E_B1_after_mask_zwfs=E_B1_before_mask_zwfs*(MFP_zwfs)
    E_C10_zwfs=MFT(NB, N, m1*(N/D),E_B1_after_mask_zwfs,inv=True,center_pixel=True)

    E_C1_zwfs=E_A1_zwfs-((1-np.exp(1j*phase))* E_C10_zwfs)

    I_R2_zwfs=np.abs(E_C10_zwfs)**2 
   
    vector_I_R2_zwfs = I_R2_zwfs.flatten()
    step_zwfs=((control_matrix_zwfs@((vector_I_R2_zwfs))))
    
    step_cubephase_zwfs.append(step_zwfs)

"""
a=[]  
while len(step_cubephase) < 10:
    a.append(step_cubephase * amp[0])
else:
    a.append(step_cubephase*amp[1])
"""
#%% sensor

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
    
cube_amp_rad=(zwfs_amp)#*((2*np.pi)/(wavelength))
plt.figure()
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



plt.plot(cube_amp_rad,cube_amp_rad,"r--",label='reponse unitaire')
plt.xlabel('Tip  applied (nm RMS) ')
plt.ylabel('Tip measured (nm RMS)')
plt.legend()
#np.save('test2_bis1.5.txt',abé_spherique)

#%%
"""

a1=np.load('test2_bis0.3.txt.npy')
a=np.load('test2_bis0.5.txt.npy')
b=np.load('test2_bis0.8.txt.npy')
c=np.load('test2_bis0.9.txt.npy')
d=np.load('test2_bis1.0.txt.npy')
e=np.load('test2_bis1.06.txt.npy')
f=np.load('test2_bis1.2.txt.npy')
plt.plot(cube_amp_rad,cube_amp_rad,label='reponse unitaire')

plt.plot(cube_amp_rad,a1,"--",label='0.3\u03BB/D')
plt.plot(cube_amp_rad,a,"--",label='0.5\u03BB/D')
plt.plot(cube_amp_rad,b,"--",label='0.8\u03BB/D')
plt.plot(cube_amp_rad,c,"--",label='0.9\u03BB/D')
plt.plot(cube_amp_rad,d,"--",label='1.0\u03BB/D')
plt.plot(cube_amp_rad,e,"--",label='1.06\u03BB/D')
plt.plot(cube_amp_rad,f,"--",label='1.2\u03BB/D')

plt.xlabel('abé spherique  applied (nm RMS) ')
plt.ylabel('abé spherique measured (nm RMS)')
plt.legend()
"""

#%%

"""
# Données d'exemple
x_sensor = cube_amp_rad 
y_sensor = tilt
# Ajuster une ligne droite (polynôme de degré 1) à vos données
coefficients = np.polyfit(x_sensor, y_sensor, 1)  # 1 pour le degré du polynôme (linéaire)
pente, ordonnee_origine = coefficients


# Créer la fonction de la droite ajustée
fit_line = pente * x_sensor + ordonnee_origine
#plt.plot(x_sensor, fit_line, "r--",color='black', label='Ajustement linéaire')
plt.legend()

equation_droite = f"y = {pente:.2f}x + {ordonnee_origine:.2f}"
print("Équation de la droite ajustée :", equation_droite)

np.save('test2_defoc21.txt',defoc)
np.save('eq21.txt',equation_droite)


#%% for create sensor response

modes2=[(1,-1,0E-9),(1,1,0E-9),(2,2,0E-9),(2,0,0E-9),(2,-2,0E-9),(3,-3,0E-9),(3,-1,0E-9),(3,1,0),(3,3,0),(4,0,0E-9)]
vecteurs_I_R2=[]
zernike_mode2=[]
for  mode2 in (modes2):
    n2, m2, amplitude_rms2 = mode2
    #pour le tip tilt amplitude = amplitude_rad/(2*np.pi)
    zernike_mode2.append(zernike_polar (theta1,rho1, amplitude_rms2, n2, m2))
    
        
zernike_mode22= zernike_mode2[0]+  zernike_mode2[1] + zernike_mode2[2]  + zernike_mode2[3]   + zernike_mode2[4]  + zernike_mode2[5]    + zernike_mode2[6]  + zernike_mode2[7]    + zernike_mode2[8]  + zernike_mode2[9]    

E_A1_temp=1*O*np.exp((1j*2*np.pi*zernike_mode22)/wavelength) #Calcul of E_A temporary in the aperture O
plt.figure('tests')
plt.clf()
plt.title('zernike_mode22')
plt.imshow(zernike_mode22)
plt.colorbar(label='Intensity')
"""
#%%
#Injection of E_A in a grill with zero padding
"""
E_A1=np.zeros((N,N),dtype="complex128")
ini=(N//2)-(D//2)
end=(N//2)+(D//2)
E_A1[ini:end,ini:end]=E_A1_temp


E_B1_before_mask=MFT(N, NB, m1*(N/D), E_A1,center_pixel=False)
E_B1_after_mask=E_B1_before_mask*(MFP)
E_C10=MFT(NB, N, m1*(N/D),E_B1_after_mask,inv=True,center_pixel=False)

E_C1=E_A1-((1-np.exp(1j*phase))* E_C10)

I_R2=np.abs(E_C1)**2    


plt.figure('test')
plt.clf()
plt.imshow(I_R2)
plt.colorbar(label='Intensity')

mean_I_R2=np.mean(I_R2) 

vector_I_R2 = I_R2.flatten()/mean_I_R2 
    
step=((control_matrix_C@((vector_I_R2))))
zer.append(zernike_mode2)

"""