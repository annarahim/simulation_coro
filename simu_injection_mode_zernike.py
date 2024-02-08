# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 13:58:59 2024

@author: arahim
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#%% VARIABLE DEFINTIONS
wavelength = 700E-6  # wavelenght in meters
D=8 # Diameter of circular aperture in meters
N=1000 #Number of pixel for zerro padding 
NA = 750 # Number of pixels along the pupil diameter D
NB = 100 # Number of pixels in the focal plane
NC=NA #Number of pixels in plane C
ND=1024 #Number of pixels in plane D
md =  20 #sampling for TF in plane D 
eps=2 #Epsilon coefficient eps=1 for Lyot corona and eps=2 for Rodier corona
m =  1.22*2 #sampling (wavelenght/D)

#%%Creation of circular aperture
circle_radius = NA/2 #radius of aperture in number of pixels
x0=(N/2)
y0=(N/2)

# creation a point grid in the xy plane
x1 = np.linspace(0, N-1, N)
y1 = np.linspace(0, N-1, N)
x, y = np.meshgrid(x1, y1)
aperture_radius = (x-x0)**2 + (y-y0)**2 <= circle_radius**2
P=np.zeros((N,N))
P[aperture_radius]=1
"""
main_rows, main_cols = N,N  # Dimensions du tableau principal
sub_rows, sub_cols = NA,NA  # Dimensions du tableau inséré

# Créer les tableaux
zerro_pad= np.zeros((main_rows, main_cols))  # Tableau principal rempli de zéros
sub_array = P  # Tableau inséré

# Calculer les décalages pour centrer le tableau inséré dans le tableau principal
start_row = (main_rows - sub_rows) // 2
start_col = (main_cols - sub_cols) // 2

# Insérer le tableau inséré au centre du tableau principal
zerro_pad[start_row:start_row+sub_rows, start_col:start_col+sub_cols] = sub_array
"""
#Circular aperture
plt.figure('aperture')
plt.title('aperture')
plt.imshow(P)
plt.colorbar(label='Intensity')
plt.show()

def zernike_polar(radius, theta, amplitude, n, m):
    rho = radius / 1.0
    if m == 0:
        return amplitude*zernike_radial(n, m, rho) * np.sqrt(n + 1)
    elif m > 0:
        return amplitude*zernike_radial(n, m, rho) * np.sqrt(2 * (n + 1)) * np.cos(m * theta)
    else:
        return amplitude*zernike_radial(n, abs(m), rho) * np.sqrt(2 * (n + 1)) * np.sin(abs(m) * theta)

def zernike_radial(n, m, rho):
    result = 0
    for k in range((n - abs(m)) // 2 + 1):
        result += (-1) ** k * np.math.factorial(n - k) / (np.math.factorial(k) * np.math.factorial((n + abs(m)) // 2 - k) * np.math.factorial((n - abs(m)) // 2 - k) ) * rho ** (n - 2 * k)
    return result

# Générer un mode de Zernike uniquement dans la région de l'ouverture circulaire
zernike_mode = np.zeros_like(P)
radius = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
theta = np.arctan2(y - y0, x - x0)
amplitude_rms=20
modes=[(1,-1,0.01)]
for mode in modes:
    n, m, amplitude_rms = mode
    amplitude = amplitude_rms / np.sqrt(n + 1)
    zernike_mode += zernike_polar(radius, theta, amplitude, n, m)

# Affichage du mode de Zernike injecté dans l'ouverture circulaire
plt.figure(6)
plt.imshow(zernike_mode)
plt.colorbar()
plt.title('Mode de Zernike (n=4, m=2) dans une ouverture circulaire')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
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

plt.figure('m')
plt.title('m')
plt.imshow(MFP)
plt.colorbar(label='Intensity')

#%%Creation of lyot stop 
LS=(circle_radius*1)# Radius of lyot stop as a function of entry aperture
radius2 = (x-x0)**2 + (y-y0)**2 <= LS**2
P_C=np.zeros((N,N))
P_C[radius2]=1  
"""
main_rows1, main_cols1 = N,N  # Dimensions du tableau principal
sub_rows1, sub_cols1 = NA,NA  # Dimensions du tableau inséré

# Créer les tableaux
zerro_pad1= np.zeros((main_rows1, main_cols1))  # Tableau principal rempli de zéros
sub_array1 = P_C  # Tableau inséré

# Calculer les décalages pour centrer le tableau inséré dans le tableau principal
start_row1 = (main_rows1 - sub_rows1) // 2
start_col1 = (main_cols1 - sub_cols1) // 2

# Insérer le tableau inséré au centre du tableau principal
zerro_pad1[start_row1:start_row1+sub_rows1, start_col1:start_col1+sub_cols1] = sub_array1
"""
plt.figure('LS')
plt.title('LS')
plt.imshow(P_C)
plt.colorbar(label='Intensity')
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
E_A=1*P*zernike_mode # electric field  plane A 
I_A=np.abs(E_A)**2 #Intensity plane A 

plt.figure('A')
plt.title('A')
plt.imshow(I_A)
plt.colorbar(label='Intensity')

# PLANE B 
#Calcul of the direct TF of the electric field at plane B 
E_B = MFT(N, NB, m, E_A)
I_B=np.abs(E_B )**2
E_B2=E_B*MFP #Multiplying the B electric field by the mask 
I_B2=np.abs(E_B2)**2

# PLANE C
#Calcul of the inverse TF of the elec field in plane B  
E_C= MFT(NB, N, m, E_B2,inv=True)

#Determination of the elec field at the C plane multiplied by the lyot stop 
E_C2=(E_A-eps*E_C)*(1-P_C)
I_C2=np.abs(E_C2)**2
plt.figure('PLANE C')
plt.title('plane C')
plt.imshow((I_C2)**0.25)
plt.colorbar(label='Intensity')

# PLANE D 
#Calcul of the TF of the elec field in plane C giving the elec field in plane D 
E_R = MFT(N,N,md,E_C2)
I_R=np.abs(E_R)**2

#Image of the PSF PLANE camera
plt.figure('PLANE CAM')
plt.title('plane CAM')
plt.imshow((I_R)**0.25)
plt.colorbar(label='Intensity')

#Determination of electric fiel without coronagraph
E_C0=E_A*(P_C)
E_D0=MFT(N,N,md,E_C0)
I_D0=np.abs(E_D0)**2
plt.figure('PLANE CAM without coronagraph')
plt.title('plane CAM without coronagraph')
plt.imshow((I_D0)**0.25)
plt.colorbar(label='Intensity')

#Determination des abérattions au plane caméra 
I=I_R/I_D0
plt.figure('rapport des I')
plt.title('rapport des I')
plt.imshow(np.log10(I))
plt.colorbar(label='Intensity')


#Determination des abérattions
I_f=(I_R-I_D0)/amplitude
plt.figure('rapport des I_f')
plt.title('rapport des I_f')
plt.imshow((I_f))
plt.colorbar(label='Intensity')

"""
# %%Champ électrique prédit à partir de la mesure d'intensité
# Fonction qui prédit la différence d'intensité en fonction des coefficients de Zernike
measured_intensity_difference=((I_R-I_D0))
zernike_modes = np.random.rand(5, 1000, 1000)
n_modes=5
def predict_intensity_difference_zernike(coefficients, zernike_modes):
    # Calcul de la différence d'intensité prédite en fonction des coefficients de Zernike
    predicted_intensity_difference = np.dot(coefficients, zernike_modes)
    return predicted_intensity_difference

# Fonction de coût pour l'optimisation
def cost_function_difference(coefficients, measured_intensity_difference, zernike_modes):
    # Prédiction de la différence d'intensité
    predicted_intensity_difference = predict_intensity_difference_zernike(coefficients, zernike_modes)
    # Calcul de la différence entre la différence d'intensité prédite et la différence d'intensité mesurée
    difference = predicted_intensity_difference - measured_intensity_difference
    # Calcul de la somme des carrés de la différence
    cost = np.sum(difference**2)
    return cost

# Coefficients initiaux de Zernike
coefficients=initial_guess = np.zeros(n_modes)

# Effectuer l'optimisation pour ajuster les coefficients de Zernike
result = minimize(cost_function_difference, initial_guess, args=(measured_intensity_difference, zernike_modes))

# Coefficients de Zernike ajustés
fitted_coefficients = result.x



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
"""