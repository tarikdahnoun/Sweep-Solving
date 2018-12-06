#Lesson 19 Homework
#Tarik Dahnoun
#Boundary Value

import numpy as np
import pylab as py

k=5.0             # k = (mu*g)/F_Rx
xi=0.0            # Initial x value
xf=1.0            # Final x value
yi=0.0            # Initial y value
yf=2.0            # Final y value
N=100             # Number of nodes
sweeps=10000     # Number of sweeps 

# Arrays to store values  
y=np.zeros(N)
y_old = np.zeros(N)

x =np.linspace(xi,xf,N)
dx=x[1]-x[0]

L1=np.zeros(sweeps)

Fx=np.zeros(N)

# First and last values of y 
y[0] = yi
y[N-1] = yf

# Trial solution 
for i in range(1, N):
    y[i] = 2.0*x[i]
        
# Relaxation Sweeps
def Relax(k,sweeps):
    for j in range(sweeps):    
        y_old = np.copy(y)
        y[1:-1] = 0.5*(y[:-2] + y[2:]) - (0.5*k*dx**2)*np.sqrt(1+(y[:-2] - y[2:])**2/(2*dx)**2)
        L1[j] = np.mean(abs(y - y_old))
    return y,L1
    
y,L1=Relax(k,sweeps)
L= sum(np.sqrt(1.+np.gradient(y,dx)**2))*dx

L=L+(1./2.)*(y[0])+(1./2.)*(y[N-1])

print "The length integral evaluates to",L

print "Error at", sweeps, "sweeps is", min(L1)

py.figure(1)
py.plot(x,y,"-",label="K value="+str(k))
# py.semilogy(L1)
py.title("Chain")
py.xlabel("x[m]", fontsize=16)
py.ylabel("y[m]", fontsize=16)
py.legend()
py.show()

py.figure(2)
K=[1,2,3,4,5,6,7,8,9,10]
for i in range(len(K)):
    Y=Relax(K[i],sweeps)[0]
    L= sum(np.sqrt(1.+np.gradient(y,dx)**2))*dx
    py.plot(K[i],L,".-")
py.title("Length by k value")
py.xlabel("K", fontsize=16)
py.ylabel("L", fontsize=16)
py.show()
    
    

