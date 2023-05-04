import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
h=1*10**(-4)#The width of the blood vessel

mu=3*10**(-3)#viscoscity of blood
a=400*10**(-9)#size of particle
U=[0.2*10**(-3),0]#inital velocity/velociy of the fluid
X=[0,U[0],h/2,U[1]]#inital conditions
M=1#mass
T=40#time factor
B=[10**(-3),10**(-3),10**(-3)]#fmag
L=10 #length of the vessel
def myboi(t,X,*args):
    return X[1]
def function(t,X,B,mu,a,U,h,M):
    print((((6*U[0])/h**2)*X[2]*(h-X[2])-X[1]),X[1]-U[0])
    time.sleep(0.5)
    return [X[1],(B[0]+6*np.pi*mu*a*(((6*U[0])/h**2)*X[2]*(h-X[2])-X[1]))/M,X[3],(B[1]-6*np.pi*mu*a*X[3])/M]
myboi.terminal = True
soly=solve_ivp(function,(0,T),X,events=myboi,args=[B,mu,a,U,h,M],rtol=1*10**(-10), atol=1*10**(-10))
#print(soly.y)
plt.plot(soly.y[0],soly.y[2])
print(soly.y_events)
plt.show()