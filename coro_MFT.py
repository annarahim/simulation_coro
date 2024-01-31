# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 14:12:37 2024

@author: arahim
"""

import numpy as np
import matplotlib.pyplot as plt

#%% VARIABLE DEFINTIONS
wavelength = 700E-6  # wavelenght in meters
D=8 # Diameter of circular aperture in meters
NA = 750 # Number of pixels along the pupil diameter D
NB = 100 # Number of pixels in the focal plane
NC=NA #Number of pixels in plane C
ND=1024 #Number of pixels in plane D
md =  20 #sampling for TF in plane D 
eps=2 #Epsilon coefficient eps=1 for Lyot corona and eps=2 for Rodier corona
m =  1.22*2 #sampling (wavelenght/D)

#%%Creation of circular aperture
circle_radius = NA/2 #radius of aperture in number of pixels
x0=(NA/2)
y0=(NA/2)

# creation a point grid in the xy plane
x1 = np.linspace(0, NA-1, NA)
y1 = np.linspace(0, NA-1, NA)
x, y = np.meshgrid(x1, y1)
aperture_radius = (x-x0)**2 + (y-y0)**2 <= circle_radius**2
P=np.zeros((NA,NA))
P[aperture_radius]=1

#%%Creation focal mask
circle_radius_MFP = (NB/2) #radius of focal mask MFP
x0_MFP=(NB/2)
y0_MFP=(NB/2)

# creation a point grid in the xy plane
x1_MFP = np.linspace(0, NB-1, NB)
y1_MFP = np.linspace(0, NB-1, NB)
x_MFP, y_MFP = np.meshgrid(x1_MFP, y1_MFP)

#Creation of ciruclar aperture 
aperture_radius = (x_MFP-x0_MFP)**2 + (y_MFP-y0_MFP)**2 <= circle_radius_MFP**2
MFP=np.zeros((NB,NB))
MFP[aperture_radius]=1

#%%Creation of lyot stop 
LS=(circle_radius*1)# Radius of lyot stop as a function of entry aperture
radius2 = (x-x0)**2 + (y-y0)**2 <= LS**2
P_C=np.zeros((NA,NA))
P_C[radius2]=1

#%%Function for calculating the semi-analytical Fourier Transform 

def MFT (NA,NB,m,E,inv=False):
    """
    NA = Input table
    NB= output tabe
    m = sampling in lamda/D
    E=Electric field of precedent plane
    inv= False by default => sign =-1 for do a TF and inv= True => sign=1 for a invers TF 
    """
    #Vectors defintinion for TF 
    xk = yk = (np.arange(NA) - NA/2)  * 1/NA
    ul = vl = (np.arange(NB) - NB/2) * m/NB
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
    Fb_UH  = (m / (NA * NB)) *F_U @E @ F_V
    
    return Fb_UH

#%% Calcul of electric field
# PLANE A
E_A=1*P # electric field  plane A 
I_A=np.abs(E_A)**2 #Intensity plane A 

# PLANE B 
#Calcul of the direct TF of the electric field at plane B 
E_B = MFT(NA, NB, m, E_A)
I_B=np.abs(E_B )**2
E_B2=E_B*MFP #Multiplying the B electric field by the mask 
I_B2=np.abs(E_B2)**2

# PLANE C
#Calcul of the inverse TF of the elec field in plane B  
E_C= MFT(NB, NA, m, E_B2,inv=True)

#Determination of the elec field at the C plane multiplied by the lyot stop 
E_C2=(E_A-eps*E_C)*(P_C)
I_C2=np.abs(E_C2)**2

# PLANE D 
#Calcul of the TF of the elec field in plane C giving the elec field in plane D 
E_D = MFT(NC,ND,md,E_C2)
I_D=np.abs(E_D)**2

#Determination of electric fiel without coronagraph
E_C0=E_A*P_C
E_D0=MFT(NC,ND,md,E_C0)
I_D0=np.abs(E_D0)**2

#D-plane intensity normalization for propagation without a coronagraph 
# Found maximum in the list
diviseur =max(max(row) for row in I_D0)
I_Dnorm=I_D/diviseur
I_D0norm=I_D0/diviseur

#%%Plot of planes 
#Circular aperture
plt.figure('aperture')
plt.title('aperture')
plt.imshow(P)
plt.colorbar(label='Intensity')
plt.show()

#Image of the PSF PLANE B before MFP 
plt.figure('plane B before mask')
plt.title('plane B before mask')
plt.imshow(np.log10(I_B))
plt.colorbar(label='Intensity') 

#Image of the mask
plt.figure('m')
plt.title('m')
plt.imshow(MFP)
plt.colorbar(label='Intensity')

#Image of the PSF PLANE B after MFP
plt.figure('plane B after MFP')
plt.title('plane B after MFP')
plt.imshow((I_B2)**0.25) #puissance 0.25 for a better display 
plt.colorbar(label='Intensity')

#Image of the PSF PLANE C before lyot stop
I_C=np.abs(E_C)**2
plt.figure('c')
plt.title('c')
plt.imshow((I_C)**0.25)
plt.colorbar(label='Intensity')

#Image of Lyot stop 
plt.figure('LS')
plt.title('LS')
plt.imshow(P_C)
plt.colorbar(label='Intensity')

#Image of the PSF PLANE C after lyot stop
plt.figure('plane C after lyot stop')
plt.title('plane C after lyot stop')
plt.imshow(I_C2)
plt.colorbar(label='Intensity')

# Intensity of plane C  (longitundinal cut)
plt.figure()
plt.title('coupe de I_C')
plt.ylabel('I_C')
abss =(np.arange(NC))*wavelength/(m*2) # Determination of abcisse in lambda/D
cut2=I_C2[NC//2,:NC]
plt.plot(abss,cut2, color='blue', alpha=1,label='plane b')

#Image of the PSF PLANE D
plt.figure('PLANE D')
plt.title('plane D')
plt.imshow((I_D)**0.25)
plt.colorbar(label='Intensity')

#Image of the PSF PLANE D normalized 
plt.figure('PLANE D normalized with max')
plt.title('plane D normalized with max')
plt.imshow(I_Dnorm)
plt.colorbar(label='Intensity')

#Image of plane D without corona
plt.figure('PLANE D0')
plt.title('plane D0')
plt.imshow((I_D0)**0.25)
plt.colorbar(label='Intensity')

#Longitdinal cut of intensity in plane D 
abcisse= ((np.arange(ND//2))/(m*md)) # Determination of abcisse in lambda/D
plt.figure()
plt.plot(abcisse,np.log10(I_D0norm[ND//2,ND//2:ND]) , color='blue', alpha=1,label='propagation without corona')
plt.plot(abcisse,np.log10(I_Dnorm[ND//2,ND//2:ND]), color='red', alpha=1,label='propagation with corona')
plt.legend()
plt.title('Cut')
plt.xlabel('$\Lambda$/D')
plt.ylabel('I_D')
plt.show()

