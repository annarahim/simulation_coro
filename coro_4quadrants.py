# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 10:41:58 2024

@author: arahim
"""
import numpy as np

import matplotlib.pyplot as plt

wavelength =  1.6E-6   # Longueur d'onde de la lumière en metres
D=8 #Diamètre de l'ouverture circulaire en m 
D = 128 # Number of pixels along the pupil diameter D
#NB = 40 # Number of pixels in the focal plane
NC=D #Nombre de pixels au plan C
N=512   #Number of pixel of zero padding 
oversize=3 # Multiple of D for the externe diameter of RLS
D_win=20 # Zoom in a windowq 20x20 for plots

#%%Creation de l'ouverture d'entrée with zero padding 
circle_radius = (D/2) #rayon de l'ouverture en nombre de pixel
x0=((N/2))#-(1/2)
y0=((N/2))#-(1/2)

# Création d'une grille de points dans le plan XY
x1 = np.linspace(0, N-1, N)
y1 = np.linspace(0, N-1, N)
x, y = np.meshgrid(x1, y1)


#Création de l'ouverture circulaire 
aperture_radius = (x-x0)**2 + (y-y0)**2 <= circle_radius**2
P=np.zeros((N,N))
P[aperture_radius]=1
"""
plt.figure('P')
plt.clf()
plt.title('P')
plt.imshow (P) 
plt.colorbar(label='Intensité')
"""
#%%Creation of cirle where the zernike is applied

circle_radius1 = 1 #radius of aperture in number of pixels
x01=0
y01=0

# creation a point grid in the xy plane
x11 = (np.arange(D)-(D/2))/(D/2)
y11 = (np.arange(D)-(D/2))/(D/2)
x111, y111= np.meshgrid(x11, y11)
aperture_radius21 =  (x111-x01)**2 + (y111-y01)**2 <= circle_radius1**2
O=np.zeros((D,D))
O[aperture_radius21]=1

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
#%%
"""
#inner diameter = cercle
LS=(circle_radius*1)# Rayon du lyot stop en fonction du rayon de l'ouverture d'entrée 
radius2 = (x-x0)**2 + (y-y0)**2 <= LS**2
P_C=np.zeros((N,N))
P_C[radius2]=1
"""
#inner diameter= square
rapport=(1)
inv_rapport=int(1/(rapport*(1/2)))
x_car0 = (np.arange (int(D*rapport))-int(D*(rapport*(1/2)))/int(D*(rapport*(1/2))))
y_car0=(np.arange (int(D*rapport))-int(D*(rapport*(1/2))))/int(D*(rapport*(1/2)))
x_car, y_car= np.meshgrid(x_car0, y_car0)
CAR = x_car+y_car <=(D)**2
CAR_bis=np.zeros((N,N))
ini3=(N//2)-(D//inv_rapport)
end3=(N//2)+(D//inv_rapport)
CAR_bis[ini3:end3,ini3:end3]=CAR
"""
plt.figure('LS')
plt.title('LS')
plt.clf()
plt.imshow(CAR_bis)
plt.colorbar(label='Intensité')
"""
#%% Creation of diameter limit for light recovery after LS 
circle_radius3 =(1) #radius of aperture in number of pixels
x03=0
y03=0

# creation a point grid in the xy plane
x13 = (np.arange(N)-(N/2))/(oversize*D/2)
y13 = (np.arange(N)-(N/2))/(oversize*D/2)
x3, y3 = np.meshgrid(x13, y13)
aperture_radius3 =  (x3-x03)**2 + (y3-y03)**2 <= circle_radius3**2
D_lim_LS=np.zeros((N,N))
D_lim_LS[aperture_radius3]=1
D_lim_LS[CAR_bis==1]=0

"""
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
modes0=[(2,0,1.76E-6) ]#1.27E-6
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
modes=[(1,-1,amp),(1,1,amp),(2,-2,amp),(2,0,(amp)),(2,2,amp),(3,-3,amp),(3,-1,amp),(3,1,amp),(3,3,amp),(4,0,amp)]
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
#%% Visualiation of abérattions
"""
Abe=np.reshape(vecteurs_I_f,(len(modes),ND,ND))
fig, axs = plt.subplots(int(np.sqrt(len(modes))+1),int(np.sqrt(len(modes))))
#fig, axs = plt.subplots(1,3)
axs[0,0].imshow(Abe[0])
axs[0,0].set_title('n=1 m=-1')
cbar = plt.colorbar(axs[0,0].images[0], ax=axs[0,0])
cbar.set_label('Intensity')

axs[0,1].imshow(Abe[1])
axs[0,1].set_title('n=1 m=1')
cbar = plt.colorbar(axs[0,1].images[0], ax=axs[0,1])
cbar.set_label('Intensity')

axs[0,2].imshow(Abe[2])                                                                                                                     
axs[0, 2].set_title('n=2 m=-2')
cbar = plt.colorbar(axs[0,2].images[0], ax=axs[0,2])
cbar.set_label('Intensity')

axs[1,0].imshow(Abe[3])
axs[1, 0].set_title('n=2 m=0')
cbar = plt.colorbar(axs[1, 0].images[0], ax=axs[1, 0])
cbar.set_label('Intensity')

axs[1,1].imshow(Abe[4])
axs[1, 1].set_title('n=2 m=2')
cbar = plt.colorbar(axs[1, 1].images[0], ax=axs[1, 1])
cbar.set_label('Intensity')

axs[1,2].imshow(Abe[5])
axs[1, 2].set_title('n=3 m=-3')
cbar = plt.colorbar(axs[1, 2].images[0], ax=axs[1, 2])
cbar.set_label('Intensity')

axs[2,0].imshow(Abe[6])
axs[2, 0].set_title('n=3 m=-1')
cbar = plt.colorbar(axs[2, 0].images[0], ax=axs[2, 0])
cbar.set_label('Intensity')

axs[2,1].imshow(Abe[7])
axs[2, 1].set_title('n=3 m=1')
cbar = plt.colorbar(axs[2, 1].images[0], ax=axs[2, 1])
cbar.set_label('Intensity')

axs[2,2].imshow(Abe[8])
axs[2, 2].set_title('n=3 m=3')
cbar = plt.colorbar(axs[2, 2].images[0], ax=axs[2, 2])
cbar.set_label('Intensity')

axs[3,0].imshow(Abe[9])
axs[3, 0].set_title('n=4 m=0')
cbar = plt.colorbar(axs[3, 0].images[0], ax=axs[3, 0])
cbar.set_label('Intensity') 
"""
#%%Création d'une nouvelle image (avec 1 ou pls modes injecter)
#%% for create sensor response
cube_amp=np.arange(-127E-9, 128E-9,((127E-9)*2/100))#range(1)
modes2=[(1,-1,cube_amp),(1,1,0E-9),(2,-2,0E-9),(2,0,0E-9),(2,2,0E-9),(3,-3,0E-9),(3,-1,0E-9),(3,1,0E-9),(3,3,0E-9),(4,0,0E-9)]
vecteurs_I_R2=[]
zernike_mode2=[]
step_cubephase=[]
sommes=[]
for i, cube in enumerate(cube_amp):
    for u, mode2 in enumerate (modes2):
        n2, m2, amplitude_rms2 = mode2
        if n2 ==1 and m2 == -1:
            
            # Utiliser une valeur différente de cube_amp
            amplitude_rms2 = cube_amp[i] 
        #pour le tip tilt amplitude = amplitude_rad/(2*np.pi)
        zernike_mode2.append(zernike_polar(theta1,rho1, amplitude_rms2, n2, m2))
    #zernike_mode22=np.array(zernike_mode2)
for t in range(0, len(zernike_mode2), 10):
    groupe_tableaux = zernike_mode2[t:t+10]
    Z=np.sum(groupe_tableaux, axis=0)
    sommes.append(Z)
for i, cube in enumerate(cube_amp):
    E_A1_temp=1*O*np.exp((1j*2*np.pi*sommes[i])/wavelength) #Calcul of E_A temporary in the aperture O

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
    vector_I_R2 = I_R2_bis.flatten()
    vecteurs_I_R2.append(vector_I_R2)
    step=((control_matrix_C@((vector_I_R2-vector_I_D0))))
    step_cubephase.append(step*(amp))

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
for liste in step_cubephase:
    tilt.append(liste[0]*((2*np.pi)/(wavelength)))
    tip.append(liste[1]*((2*np.pi)/(wavelength)))
    astig_oblique.append(liste[2]*((2*np.pi)/(wavelength)))
    defoc.append(liste[3]*((2*np.pi)/(wavelength)))
    astig_vertical.append(liste[4]*((2*np.pi)/(wavelength)))
    trefoil_vertical.append(liste[5]*((2*np.pi)/(wavelength)))
    coma_vertical.append(liste[6]*((2*np.pi)/(wavelength)))
    coma_horizontal.append(liste[7]*((2*np.pi)/(wavelength)))
    trefoil_oblique.append(liste[8]*((2*np.pi)/(wavelength)))
    abé_spherique.append(liste[9]*((2*np.pi)/(wavelength)))
"""
np.save('Simu7.txt',defoc)
a=np.load('Simu.txt.npy')
b=np.load('Simu2.txt.npy')
c=np.load('Simu3.txt.npy')
d=np.load('Simu4.txt.npy')
e=np.load('Simu5.txt.npy')
f=np.load('Simu6.txt.npy')
g=np.load('Simu7.txt.npy')
plt.plot(cube_amp,a,label='defoc_sensor=-2rad RMS')
plt.plot(cube_amp,b,label='defoc_sensor=2rad RMS')
plt.plot(cube_amp,c,label='defoc_sensor=-5rad RMS')
plt.plot(cube_amp,d,label='defoc_sensor=5rad RMS')
plt.plot(cube_amp,e,label='defoc_sensor=10rad RMS')
plt.plot(cube_amp,f,label='defoc_sensor=15rad RMS')
plt.plot(cube_amp,g,label='defoc_sensor=20rad RMS')

"""
cube_amp_rad=(cube_amp)*((2*np.pi)/(wavelength))
plt.figure()
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

plt.xlabel('Amplitude of astig_vertical applied (radian RMS)')
plt.xlim(-0.6,0.6)
plt.ylabel('astig_vertical measured  by the sensor (radian RMS)')
plt.ylim(-0.6,0.6)


# Données d'exemple
x_sensor = cube_amp_rad
y_sensor = tilt
# Ajuster une ligne droite (polynôme de degré 1) à vos données
coefficients = np.polyfit(x_sensor, y_sensor, 1)  # 1 pour le degré du polynôme (linéaire)
pente, ordonnee_origine = coefficients

# Créer la fonction de la droite ajustée
fit_line = pente * x_sensor + ordonnee_origine
plt.plot(x_sensor, fit_line, color='black', label='Ajustement linéaire')
plt.legend()

equation_droite = f"y = {pente:.2f}x + {ordonnee_origine:.2f}"
print("Équation de la droite ajustée :", equation_droite)

#%%

#%%Correction 

"""    
S=[]
coeff=(step)*(amp/1E-9)
for k, mode2 in enumerate(modes2):
    
    if coeff[k]>(max(coeff/10)): 
   
        S.append(coeff[k]*Map[k])
    else :
        S.append(0*Map[k])


correction = S[0]+S[1]+S[2]+S[3]+S[4]+S[5]+S[6]+S[7]+S[8]+S[9]

cor=np.zeros((N,N))#,dtype="complex128")
ini=(N//2)-(D//2)
end=(N//2)+(D//2)
cor[ini:end,ini:end]=correction
zernike_mode222=np.zeros((N,N))#,dtype="complex128")
zernike_mode222[ini:end,ini:end]=sommes[0]

I_A1=np.abs(E_A1)
current_phasemap=zernike_mode222
Actuator=(current_phasemap-(cor))
act=sommes[0]-correction

Phi_A_temp=1*O*np.exp((1j*2*np.pi*(sommes[0]+act))/wavelength)

Phi_A=np.zeros((N,N),dtype="complex128")
ini=(N//2)-(D//2)
end=(N//2)+(D//2)
Phi_A[ini:end,ini:end]=Phi_A_temp

Phi_B1_before_mask=MFT(N, N, N, Phi_A,center_pixel=True)
Phi_B1_after_mask=Phi_B1_before_mask*MFP 

Phi_C1_before_LS=MFT(N, N, N,Phi_B1_after_mask,inv=True,center_pixel=True)
Phi_C1_after_LS=(Phi_C1_before_LS)*D_lim_LS*np.exp((1j*2*np.pi*zernike_mode4)/wavelength)
Phi_R2=MFT(N, N, N,Phi_C1_after_LS,center_pixel=True)
new=np.abs(Phi_R2)**2
new_bis=new[ini1:end1,ini1:end1]
vector_new = new_bis.flatten()
step2=((control_matrix_C@((vector_new-vector_I_D0))))


plt.figure('current_phasemap')
plt.title('current_phasemap')
plt.imshow(sommes[0])
plt.colorbar(label='Intensité') 


plt.figure('correct_phasemap')
plt.title('correct_phasemap')
plt.imshow(correction)
plt.colorbar(label='Intensité')


plt.figure('Actuator')
plt.title('Actuator')
plt.imshow(act)
plt.colorbar(label='Intensité') 

plt.figure('Image abberated')
plt.title('Image abberated')
plt.imshow(I_R2_bis)
plt.colorbar(label='Intensité') 

plt.figure('Image corrected')
plt.title('Image corrected')
plt.imshow(new_bis)
plt.colorbar(label='Intensité') 



plt.figure('Image without aberattion')
plt.title('Image without abérration')
plt.imshow(I_D0_bis)
plt.colorbar(label='Intensité') 
"""
#%% PLAN A
"""
E_A=1*P#*np.exp((1j*2*np.pi*zernike_mode22)/wavelength) # champ electrique plan A 
I_A=np.abs(E_A)**2 #Intensité plan A 
plt.colorbar(label='Intensité')   
plt.figure('ouverture')
plt.title('ouverture')
plt.imshow(P)
plt.colorbar(label='Intensité')
plt.show()
E_B0_before_mask=MFT(ND, ND, ND, E_A,center_pixel=True)
I_B=np.abs(E_B0_before_mask )**2
E_B2=E_B0_before_mask*(MFP) #Multiplication du champ B par le masque 
I_B2=np.abs(E_B2)**2
E_C=MFT(ND, ND, ND, E_B2,inv=True,center_pixel=True)
I_C=np.abs(E_C)**2
E_C2=(E_C)*(P_C)
I_C2=np.abs(E_C2)**2
E_D=MFT(ND, ND, ND, E_C2,center_pixel=True)
I_D=np.abs(E_D)**2


plt.figure('10')
plt.title('plan B avant masque')
plt.imshow((I_B)**0.25)
plt.colorbar(label='Intensité') 
plt.figure('plan B apres MFP')
plt.title('plan B apres MFP')
plt.imshow((I_B2)**0.25) #puissance 0.25 pour un meilleur affichage
plt.colorbar(label='Intensit"é')
plt.figure('c')
plt.title('c')
plt.imshow((I_C)**0.25)
plt.colorbar(label='Intensité')
plt.figure('PLAN D')
plt.title('plan D')
plt.imshow((I_D)**0.25)
plt.colorbar(label='Intensité')
"""

"""

"""
#%% Simulation avec la fft
#PLAN B 
#Calcul de la TF direct du champ elec au plan B 
"""
E_B = np.fft.fftshift(np.fft.fft2((E_A),norm='ortho'))
I_B=np.abs(E_B )**2
plt.figure('plan B avant masque')
plt.title('plan B avant masque')
plt.imshow((I_B)**0.25)
plt.colorbar(label='Intensité')   


E_B2=E_B*(MFP) #Multiplication du champ B par le masque 
I_B2=np.abs(E_B2)**2
plt.figure('plan B apres MFP')
plt.title('plan B apres MFP')
plt.imshow((I_B2)**0.25) #puissance 0.25 pour un meilleur affichage
plt.colorbar(label='Intensit"é')

#PLAN C
#Calcul de la TF inverse du champ elec du plan B  
E_C= (np.fft.ifft2((E_B2),norm='ortho'))
I_C=np.abs(E_C)**2
plt.figure('c')
plt.title('c')
plt.imshow((I_C)**0.25)
plt.colorbar(label='Intensité')

#Determination du champ elec au plan C multiplier par le lyot stop 
E_C2=(E_C)*(P_C)
I_C2=np.abs(E_C2)**2
plt.figure('plan C apres lyot stop')
plt.title('plan C apres lyot stop')
plt.imshow(I_C2**0.25)
plt.colorbar(label='Intensité')


#frequence2 = (np.arange(NC))*wavelength/(m*2)
coupes3=I_C[ND//2,:ND]
plt.figure('verif')
plt.plot(np.log10(coupes3), color='blue', alpha=1,label='plan b')

#PLAN D 
#Calcul de la TF du champ elec au plan C donnant le champ elec au plan D 
E_D =(np.fft.fft2(np.fft.fftshift(E_C2),norm='ortho'))
I_D=np.abs(E_D)**2
plt.figure('PLAN D')
plt.title('plan D')
plt.imshow((I_D)**0.25)
plt.colorbar(label='Intensité')


max_IB=np.max(I_B)
coupes2=I_D[ND//2,ND//2:ND]/max_IB
coupes1=I_B[ND//2,ND//2:ND]/max_IB
plt.figure(' D')
plt.plot(np.log10(coupes2), color='blue', alpha=1,label='plan d')

plt.plot(np.log10(coupes1), color='red', alpha=1,label='plan b')
"""
#%%

