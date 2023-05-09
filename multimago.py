import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import concurrent.futures
import time
def BBoi(B):
    h=25*10**(-3)#The width of the blood vessel
    myguy=np.linspace(0,h,5)
    holdyboiy=[]
    holdyboix=[]
    holdyboivol=[]
    for i in myguy:
        mu=3*10**(-3)#viscoscity of blood
        a=400*10**(-9)#size of particle
        U=[2*10**(-3),0]#inital velocity/velociy of the fluid
        X0=[0,U[0],i*1/2,U[1]]#inital conditions
        M=1#mass
        T=100#time factor
        #B=[10**(-6),10**(-4),10**(-3)]#fmag
        L=5*10**(-2) #length of the vessel
        print(i)
        def myboi(t,X,*args):
            #print(X[2]-1,X[2])
            
            return X[2]-1
        def function(t,X,B,mu,a,U,h,M,L):
            #print((L**(2)*B[0])/(M*U[0]**(2)*h)+((6*np.pi*mu*a*L)/(M*U[0]))*(((6)*X[2]*(1-X[2])-X[1])))
            #print(X[1])#,X[2])
            #time.sleep(0.1)
            return [X[1],(L**(2)*B[0])/(M*U[0]**(2)*h)+((6*np.pi*mu*a*L)/(M*U[0]))*(((6)*X[2]*(1-X[2])-X[1])),X[3],(L**(2)*B[1])/(M*U[0]**(2)*h)-((6*np.pi*mu*a*L)/(M*U[0]))*X[3]]
        myboi.terminal = True
        soly=solve_ivp(function,(0,T),X0,events=myboi,args=[B,mu,a,U,h,M,L],rtol=1*10**(-5), atol=1*10**(-5),max_step=10**(-3))
        #print(soly.y[3,-1])
        holdyboix.append(soly.y[0])
        holdyboiy.append(soly.y[2])
        holdyboivol.append(soly.y[1])
        if i-h/2==0:
            plt.plot(L*soly.y[0],h*soly.y[2])
            
            hchy=np.zeros(len(soly.y[0]))
            hchy=hchy+h
            plt.plot(L*soly.y[0],hchy)
            #print(soly.y_events)
            plt.savefig(f"plots/{B}.png")
            plt.cla()
            plt.plot(L*soly.y[0],soly.y[1])
            plt.savefig(f"plots/{B}vol.png")
            plt.cla()
    #print(h)
    for i in range(len(holdyboiy)):
        #print(holdyboiy[i])
        
        plt.plot(L*holdyboix[i],h*holdyboiy[i])
          
        hchy=np.zeros(len(soly.y[0]))
        hchy=hchy+h
    plt.plot(L*holdyboix[i],hchy)
    
          #print(soly.y_events)
    plt.savefig(f"plots/{B}multi.png")
    plt.cla()
    for i in range(len(holdyboiy)):
        plt.plot(L*holdyboix[i],holdyboivol[i])
    plt.savefig(f"plots/{B}multivol.png")
    plt.cla()
trils=1   
k=np.ones((trils,2))
for i in range(trils):
    k[i]=k[i]*10**(i-9)
BBoi(k[i])
if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        secs = [[10**(-6),10**(-4),10**(-3)]]
        results = executor.map(BBoi, k)
