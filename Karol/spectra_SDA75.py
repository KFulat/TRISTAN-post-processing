import numpy as np
import matplotlib.pyplot as plt
import scipy as scp
import h5py
from scipy import ndimage
from scipy.optimize import curve_fit
from numba import jit


def grfunct_pow(X,A,F, *argv):
    F=np.log10(A[0])+A[1]*X
    if argv != []:
        argv[0]=[[1.0/A[0]+X], [X]]

def grfunct_ekin(X,A,F, *argv):
    bx=np.exp(-A[1]*X)
    F=A[0]*np.sqrt(X)*bx
    if argv != []:
        argv[0]=[[np.sqrt(X)*bx], [-A[0]*X*np.sqrt(X)*bx]]

def grfunct_ekin_log(X,A0,A1,F, *argv):
    bx=np.exp(-A1*X)
    return A0*X*np.sqrt(X)*bx
    if argv != []:
        argv[0]=[[X*np.sqrt(X)*bx], [-A[0]*X*X*np.sqrt(X)*bx]]

def string_format(a):
    if (a < 10):
        b = str("{0:1.0f}".format(a))
    if (a >= 10 and a < 100):
        b = str("{0:2.0f}".format(a))
    if (a >= 100 and a < 1000):
        b = str("{0:3.0f}".format(a))
    if (a >= 1000 and a < 10000):
        b = str("{0:4.0f}".format(a))
    if (a >= 10000 and a <100000):
        b = str("{0:5.0f}".format(a))
    if (a >= 100000 and a < 1000000):
        b = str("{0:6.0f}".format(a))
    return b

#simulation parameters
c = 0.5 #speed of light
me = 1.0
mi = 100.0
mie = mi/me

vjet = 0.1*c
vth = 0.387298*c

theta = 75

b0 = 0.5491
b0x = b0*np.cos(theta*np.pi/180)
b0y = b0*np.sin(theta*np.pi/180)
b0z = 0.0

r = 3.8 #compression factor
phi = 0.48*mi/me*(vjet/c)**2 #shock potential

scat = 0.0 #scattering efficiency
order = 2 #order of scattering (1 or 2)

#de Hoffman-Teller (HT) frame
vsh = 0.15*c
vt = vsh/np.cos(theta*np.pi/180)
gamt = c/np.sqrt(c**2 - vt**2)
print('vt= ',vt)

#scaling
gamma = 16329.956
Lsi = 100.0

#random distribution generation
parts = 1000000 # 200000000L #number of particles
rrr = 300000 #should be no less then number of reflected particles (k=0.029644 at theta=65)

#seed values
seed1 = 100
seed2 = 200
seed3 = 300
seed4 = 400

#momentum arrays
px = np.zeros(parts)
py = np.zeros(parts)
pz = np.zeros(parts)


if scat != 0:
    pp = np.zeros(rrr)
    pt = np.zeros(rrr)

print('momentum arrays created')

sig = vth/np.sqrt(2.0)
sig = sig/np.sqrt(1.0 - (vth/c)**2)

r1 = np.sqrt(-2.0*np.log(1.0 - np.random.rand(parts)))
r2 = 2.0*np.pi*np.random.rand(parts)
px = sig*r1*np.cos(r2)
py = sig*r1*np.sin(r2)

r1 = np.sqrt(-2.0*np.log(1.0 - np.random.rand(parts)))
r2 = 2.0*np.pi*np.random.rand(parts)
pz = sig*r1*np.cos(r2)

print('random momenta generated')

#spectra parameters
gPS = 500
pmin = 0.00001
pmax = 1000.0

dp=np.log10(pmax/pmin)/gPS

#distribution arrays initial
distr1 = np.zeros(gPS)
spctr1 = np.zeros(gPS)

#distribution arrays reflected
distr2 = np.zeros(gPS)
spctr2 = np.zeros(gPS)

#distribution arrays scattered and again reflected
distr3 = np.zeros(gPS)
spctr3 = np.zeros(gPS)

#distribution arrays secondary scattered and again reflected
distr4 = np.zeros(gPS)
spctr4 = np.zeros(gPS)

rr = 0 #counter of the reflected particles
ss = 1
tt = 1
gmax = 0.0 #maximum achieved energy
dgam = 0.0 #absolute SDA acceleration
dgamr = 0.0 #relative SDA acceleration

dpart = 1000000 #code progress indicator

@jit(nopython=True)
def par_step(i,px,py,pz,distr1,distr2):
    rr = 0 #counter of the reflected particles
    ss = 1
    tt = 1
    gmax = 0.0 #maximum achieved energy
    dgam = 0.0 #absolute SDA acceleration
    dgamr = 0.0 #relative SDA acceleration

    dpart = 1000000 #code progress indicator
    if (i+1)%dpart == 0:
        print('processed particles =',(i+1)/dpart,' mln')

    p = np.sqrt(px[i]**2 + py[i]**2 + pz[i]**2)

    gam = np.sqrt(p**2 + c**2)/c
    p = p*gam**0.176 #0.3

    cosa = (px[i]*b0x + py[i]*b0y)/(p*b0)
    pparl = p*cosa
    pperp = p*np.sqrt(1.0 - cosa**2)

    gam1 = np.sqrt(p**2 + c**2)/c
    ekin1 = (gam1 - 1.0)*me*c**2
    gg = np.int(np.round(np.log10(ekin1/pmin)/dp))
    if gg >= 0 and gg < gPS:
        distr1[gg] = distr1[gg] + 1

    gam2 = 0.0

    #reflection
    vparl = pparl/gam1
    vperp = pperp/gam1

    if vparl < vt:
        ght = gam1*gamt*(1.0 - vparl*vt/c**2)
        tana = np.tan(np.arcsin((1.0/r*((ght + phi)**2 - 1.0)/(ght**2 - 1.0))**0.5))
        if vperp >= gamt*(vt - vparl)*tana:
            rr = rr + 1
            gam2 = gam1*(2.0*vt*(vt - vparl)/(c**2 - vt**2) + 1.0)
            #if scat != 0:
            #    pp[rr-1] = c*np.sqrt(gam2**2 - 1.0)

    ekin2 = (gam2 - 1.0)*me*c**2

    #gg = np.int(np.round(np.log10(np.absolute(ekin2/pmin))/dp))
    gg = np.int(np.round(np.log10(ekin2/pmin)/dp))

    if gg >= 0 and gg < gPS:
        distr2[gg] = distr2[gg] + 1
    if gam2 >= gmax:
        gmax = gam2
    if (gam2 - gam1) >= dgam:
        dgam = (gam2 - gam1)
    if (gam2 - gam1)/gam1 >= dgamr:
        dgamr = (gam2 - gam1)/gam1

    return px,py,pz,distr1,distr2


for i in range(0,parts):
    px,py,pz,distr1,distr2=par_step(i,px,py,pz,distr1,distr2)

if scat != 0:
    #;;;; 1 ;;;;;
    px = np.zeros(rrr)
    py = np.zeros(rrr)
    pz = np.zeros(rrr)

    print('new random momenta generated')

    #scattering of the reflected particles
    r5 = 2.0*np.pi*np.random.uniform(rrr)
    px = pp*np.cos(r5)
    py = pp*np.sin(r5)
    r6 = 2.0*np.pi*np.random.uniform(rrr)
    pz = pp*np.cos(r6)

    print('scattering applied')

    ss = ss - 1
    for i in range(0, rrr):
        cosa = (px[i]*b0x + py[i]*b0y)/(pp[i]*b0)
        pparl = pp[i]*cosa
        pperp = pp[i]*np.sqrt(1.0 - cosa**2)
        gam1 = np.sqrt(pp[i]**2 + c**2)/c
        gam2 = 0.0
        #reflection
        vparl = pparl/gam1
        vperp = pperp/gam1
        if vparl < vt:
            ght = gam1*gamt*(1.0 - vparl*vt/c**2)
            tana = np.tan(np.arcsin((1.0/r*((ght + phi)**2 - 1.0)/(ght**2 - 1.0))**0.5))
            if vperp > gamt*(vt - vparl)*tana:
                ss = ss + 1
                gam2 = gam1*(2.0*vt*(vt - vparl)/(c**2 - vt**2) + 1.0)
                pt[ss-1] = c*np.sqrt(gam2**2 - 1.0)
        ekin = (gam2 - 1.0)*me*c**2
        gg = np.round(np.log10(ekin/pmin)/dp)
        if gg > 0 and gg < gPS:
            distr3[gg] = distr3[gg] + 1

    if order > 1:
        #;;;; 2 ;;;;;
        px[:] = 0.0
        py[:] = 0.0
        pz[:] = 0.0

        print('new random momenta generated 2-nd')

        #scattering of the reflected particles
        r7 = 2.0*np.pi*np.random.uniform(rrr)
        px = pt*np.cos(r7)
        py = pt*np.sin(r7)
        r8 = 2.0*np.pi*np.random.uniform(rrr)
        pz = pt*np.cos(r8)

        print('scattering applied 2-nd')

        tt = tt - 1
        for i in range(0, rrr):
            cosa = (px[i]*b0x + py[i]*b0y)/(pt[i]*b0)
            pparl = pt[i]*cosa
            pperp = pt[i]*np.sqrt(1.0 - cosa**2)
            gam1 = np.sqrt(pt[i]**2 + c**2)/c
            gam2 = 0.0
            #reflection
            vparl = pparl/gam1
            vperp = pperp/gam1
            if vparl < vt:
                ght = gam1*gamt*(1.0 - vparl*vt/c**2)
                tana = np.tan(np.arcsin((1.0/r*((ght + phi)**2 - 1.0)/(ght**2 - 1.0))^0.5))
                if vperp > gamt*(vt - vparl)*tana:
                    tt = tt + 1
                    gam2 = gam1*(2.0*vt*(vt - vparl)/(c**2 - vt**2) + 1.0)

            ekin = (gam2 - 1.0)*me*c**2
            gg = np.round(np.log10(ekin/pmin)/dp)
            if gg > 0 and gg < gPS:
                distr4[gg] = distr4[gg] + 1

at = np.sum(distr1)

spctr1 = distr1/(at*dp)
spctr2 = distr2/(at*dp) #*(1.0 - scat*ss/rr)
spctr3 = distr3/(at*dp)*scat*(1.0 - scat**2*tt/ss)
spctr4 = distr4/(at*dp)*scat**2
spctr = spctr1 + spctr2 #+ spctr3 + spctr4

refl = 1.0*rr/parts
print('refl =', refl)
print('gmax =', gmax)
print('dgam =', dgam)
print('dgamr =', dgamr)

refls=string_format(refl)
gmaxs=string_format(gmax)
dgams=string_format(dgam)
dgamrs=string_format(dgamr)

xp=10**(np.log10(pmin)+np.arange(gPS)*dp)

plt.loglog(xp*4,spctr)
plt.axis([1e-2,100.0,1e-6,3])

#RELATIVISTIC Maxwellian fit
datafit=1 #set to 1 to fit
if datafit==1:
    dp0fit=0.02
    dp1fit=0.25
    B=[1000.0,50.0]

    is1=np.int32(np.round(np.log10(dp0fit/pmin)/dp))
    is2=np.int32(np.round(np.log10(dp1fit/pmin)/dp))

    Y=spctr[is1:is2]
    xpfit=xp[is1:is2]
    weights=1.0/Y

    B,yfit=curve_fit(grfunct_ekin_log,xpfit,Y)
    print("B ekin fit ",B)
    xpfit1=xp
    yf=B[0]*np.sqrt(xpfit1)*np.exp(-B[1]*1.0*xpfit1)
    print("fit")
    plt.plot(xp*4,xpfit1*yf, linestyle='--')
    #plt.text(0.1,0.001,"Maxwellian fit",fontsize=14,fontweight=0)
plt.show()

with open("SDA75.txt", 'w') as f:
    for item in spctr:
        f.write("%s\n" % item)
f.close()

