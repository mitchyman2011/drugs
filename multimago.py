import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import concurrent.futures
import time
import os
def BBoi(B):
    names=["artery","arteriole","capillary","venule","vein"]
    hes=[3*10**(-3),3*10**(-5),7*10**(-6),4*10**(-5),5*10**(-3)]
    muy=[4.5*10**(-3),4.5*10**(-3),4.5*10**(-3),4.5*10**(-3),4.5*10**(-3)]
    a=400*10**(-9)#size of particle
    uy=[[1*10**(-1),0],[1*10**(-2),0],[7*10**(-4),0],[4*10**(-3),0],[1*10**(-1),0]]
    Ls=[1*10**(-1),7*10**(-4),6*10**(-4),8*10**(-4),1*10**(-1)]
    for jef in range(len(names)):
        
        h=hes[jef]#25*10**(-3)#The width of the blood vessel
        mu=muy[jef]#3*10**(-3)#viscoscity of blood
        U=uy[jef]#[2*10**(-3),0]#inital velocity/velociy of the fluid
        M=1#mass
        L=Ls[jef]#5*10**(-2) #length of the vessel
        T=100#time factor
        myguy=np.linspace(0,h,30)
        holdyboiy=[]
        holdyboix=[]
        holdyboivol=[]
        holdyboivolx=[]
        for i in myguy:
            X0=[0,U[0],i/h,U[1]]#inital conditions
            
           

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
            soly=solve_ivp(function,(0,T),X0,events=myboi,args=[B,mu,a,U,h,M,L],rtol=1*10**(-5), atol=1*10**(-5),max_step=10**(-1))
            #print(soly.y[3,-1])
            holdyboix.append(soly.y[0])
            holdyboiy.append(soly.y[2])
            holdyboivol.append(soly.y[3])
            holdyboivolx.append(soly.y[1])
            if i-h/2==0:
                plt.plot(L*soly.y[0],h*soly.y[2])
                
                hchy=np.zeros(len(soly.y[0]))
                hchy=hchy+h
                plt.plot(L*soly.y[0],hchy)
                plt.title(f"Path of a particle in the {names[jef]} starting at half height")
                plt.ylabel("The height in The vessel(m)")
                plt.xlabel("Distance along the vessel (m)")
                
                #print(soly.y_events)
                plt.savefig(f"{names[jef]}/{names[jef]}{B}.png")
                plt.cla()
                plt.title(f"Velocity of a particle along the {names[jef]} starting at half height")
                plt.ylabel("Velocity in the hight direction (m/s)")
                plt.xlabel("Distance along the vessel (m)")
                
                plt.plot(L*soly.y[0],soly.y[3])
                plt.savefig(f"{names[jef]}/{names[jef]}{B}vol.png")
                plt.cla()
        #print(h)
        for i in range(len(holdyboiy)):
            #print(holdyboiy[i])
            
            plt.plot(L*holdyboix[i],h*holdyboiy[i])
              
            hchy=np.zeros(len(soly.y[0]))
            hchy=hchy+h
        plt.plot(L*holdyboix[i],hchy)
        plt.title(f"Path of the particle through the {names[jef]}")
        plt.ylabel("The height in the vessel (m)")
        plt.xlabel("Distance along the vessel (m)")
        
              #print(soly.y_events)
        plt.savefig(f"{names[jef]}/{names[jef]}{B}multi.png")
        plt.cla()
        for i in range(len(holdyboiy)):
            plt.plot(L*holdyboix[i],holdyboivol[i])
        plt.title(f"Velocity of the particles in the height of the {names[jef]}")    
        plt.ylabel("Velocity in the hight direction (m/s)")
        plt.xlabel("Distance along the vessel (m)")
        plt.savefig(f"{names[jef]}/{names[jef]}{B}multivol.png")
        plt.cla()
        for i in range(len(holdyboiy)):
            plt.plot(L*holdyboix[i],holdyboivolx[i])
        plt.title(f"Velocity of the particles along the {names[jef]}")
        plt.ylabel("Velocity along the vessel (v/m)")
        plt.xlabel("Distance along the vessel (m)")
        plt.savefig(f"{names[jef]}/{names[jef]}{B}multivolx.png")
        plt.cla()
trils=12
frills=12
f=0
k=np.ones((trils*frills,2))
for j in range(frills):
   for i in range(trils):
       print(f,j)
       k[f,1]=k[f,1]*10**(j-12)
       k[f,0]=k[f,0]*10**(i-12)
       f=f+1

print(k)
if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        secs = [[10**(-6),10**(-4),10**(-3)]]
        results = executor.map(BBoi, k)
