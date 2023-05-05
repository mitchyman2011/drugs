import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
h=25*10**(-3)#The width of the blood vessel

mu=3*10**(-3)#viscoscity of blood
a=400*10**(-9)#size of particle
U=[2*10**(-2),0]#inital velocity/velociy of the fluid
X=[0,U[0],1/2,U[1]]#inital conditions
M=1#mass
T=100#time factor
B=[10**(-5),10**(-5),10**(-3)]#fmag
L=5*10**(-2) #length of the vessel
def myboi(t,X,*args):
    #print(X[2]-1,X[2])
    
    return X[2]-1
def function(t,X,B,mu,a,U,h,M,L):
    #print((L**(2)*B[0])/(M*U[0]**(2)*h)+((6*np.pi*mu*a*L)/(M*U[0]))*(((6)*X[2]*(1-X[2])-X[1])))
    print(X[1])#,X[2])
    #time.sleep(0.1)
    return [X[1],(L**(2)*B[0])/(M*U[0]**(2)*h)+((6*np.pi*mu*a*L)/(M*U[0]))*(((6)*X[2]*(1-X[2])-X[1])),X[3],(L**(2)*B[0])/(M*U[0]**(2)*h)-((6*np.pi*mu*a*L)/(M*U[0]))*X[3]]
myboi.terminal = True
soly=solve_ivp(function,(0,T),X,events=myboi,args=[B,mu,a,U,h,M,L],rtol=1*10**(-5), atol=1*10**(-5),max_step=10**(-3))
print(soly.t[-1])
plt.plot(L*soly.y[0],h*soly.y[2])

hchy=np.zeros(len(soly.y[0]))
hchy=hchy+h
plt.plot(L*soly.y[0],hchy)
print(soly.y_events)
plt.show()
plt.plot(L*soly.y[0],soly.y[1])
plt.show()