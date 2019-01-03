#Chua Encryption for Runge-Kutta 4th Order
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import sys

c0 = 15.6
c1 = 1.0
c2 = 28.0
m0 = -1.143
m1 = -0.714

def f(x):
    f = m1*x+(m0-m1)/2.0*(abs(x+1.0)-abs(x-1.0))
    return f

def dH_dt(H, t=0):
    return np.array([c0*(H[1]-H[0]-f(H[0])),
                  c1*(H[0]-H[1]+H[2]),
                  -c2*H[1]])

alphabet = [i for i in range(1,44)]
phrase = raw_input("Message: ").lower()
encrypt = [0 for i in range(0, len(phrase))]
for i in range(0, len(phrase)):
    if (ord(phrase[i]) >= 97 and ord(phrase[i]) <= 122):
        encrypt[i] = alphabet[ord(phrase[i]) - 97];
    elif (ord(phrase[i]) >= 48 and ord(phrase[i]) <= 57):
        encrypt[i] = alphabet[ord(phrase[i]) - 43];
    elif (ord(phrase[i]) == 32):
        encrypt[i] = alphabet[36];
    elif (ord(phrase[i]) == 33):
        encrypt[i] = alphabet[37];
    elif (ord(phrase[i]) == 40):
        encrypt[i] = alphabet[38];
    elif (ord(phrase[i]) == 41):
        encrypt[i] = alphabet[39];
    elif (ord(phrase[i]) == 46):
        encrypt[i] = alphabet[40];
    elif (ord(phrase[i]) == 58):
        encrypt[i] = alphabet[41];
    elif (ord(phrase[i]) == 59):
        encrypt[i] = alphabet[42];
    elif (ord(phrase[i]) == 63):
        encrypt[i] = alphabet[43];
print encrypt
dt = 0.001
dx = 1.0/44.0
x0 = random.random();
y0 = random.random();
z0 = random.random();
encrypt2 = [0 for i in range(0,len(phrase))];
for i in range(0, len(phrase)):
    tf = dt * (i + 1) / (encrypt[i] * dx)
    H0 = [x0, y0, z0]
    t = np.linspace(0, tf, 1000)
    H, infodict = integrate.odeint(dH_dt, H0, t, full_output=True)
    encrypt2[i] = H[len(H) - 1,1]
print encrypt2;
decrypt = np.zeros(len(phrase))
index = np.zeros(len(phrase))
for i in range(0,1):
    tf = dt * (i + 1) / dx
    t = np.linspace(0, tf, 1000)
    H,infodict = integrate.odeint(dH_dt, H0, t, full_output=True);
    diff = [0 for i in range(0, len(H))]
    for j in range(1,len(H)):
        if(encrypt2[i] - H[j,1] < .025 and encrypt2[i] - H[j,1] > -.025):
            diff[j] = 1/(t[j] * dx / (dt * i));
        else:
            diff[j] = sys.maxint;
    print diff
    for j in range(0,len(H)):
        if (abs(diff[j] - int(diff[j])) >= .1):
            abs(diff[j] - int(diff[j]))
            diff[j] = sys.maxint;
    print diff
    H, index[i] = min(diff);
    new_diff = sorted(diff);
    elem = [i for i in range(0, len(new_diff)) if (new_diff[i] < sys.maxint)];
    decrypt[i] = H;
print decrypt
