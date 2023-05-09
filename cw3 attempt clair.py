import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
#Mass small, so d2/dtstar2 terms much smaller than other terms so can discount them
#Trial for modelling - using random numbers for now
# L = 10
# h = 5
# Fmagx = 10
# Fmagy = 10
# mu = 20
# a = 15
# U0 = 25


#more realistic values:
#artery
# L = 1e-1 # from giles paper
# h = 3e-3 # from giles paper
# Fmagx = 1e-7 #need to check
# Fmagy = 1e-7 #need to check
# mu = 3e-3 # from giles paper
# a = 2.5e-9
# U0 = 1e-1 # from giles paper

L = 600e-6
h = 30e-6
Fmagx = 1e-7
Fmagy = 1e-7
mu = 4.5e-3
a = 25e-4
U0 = 0.26

#To match mago2 file:
# L = 5e-2
# h = 25e-3
# Fmagx = 1e-6
# Fmagy = 1e-6
# mu = 3e-3
# a = 400e-9
# U0 = 2e-4

def rhs(t, q, L, Fmagx, Fmagy, mu, a, h, U0):
    """
    Function to generate the RHS of the initial value ODE system to solve

    Parameters
    ----------
    t : array
        the values of time over which to solve the IVP
    q : array
        the vector [Xstar, Ystar]
    L : float
        length of vessel
    Fmagx : float
        magnetic force on particle in x direction
    Fmagy : float
        magnetic force on particle in y direction
    mu : float
        fluid viscosity
    a : float
        particle radius
    h : float
        height of vessel
    U0 : float
        average velocity along the channel

    Returns
    -------
    dqdtstar : array
        right hand side of the system which is needed to pass into solve_ivp

    """
    #Initialise the vector dqdtstar as having the same size as q but entries are 0
    dqdtstar = np.zeros_like(q)
    #Unpack the vector q
    Xstar = q[0]
    Ystar = q[1]
    #Assign values of dqdtstar
    dqdtstar[0] = Fmagx/(6 * np.pi * mu * a * U0) + 6 * Ystar * (1 - Ystar)
    dqdtstar[1] = L * Fmagy / (6 * np.pi * mu * a * h * U0)
    #return RHS of the system
    return dqdtstar

#Define values of time over which to solve the ODE
tvals = np.linspace(0, 15, 1000)

#Solve the IVP 
sol = solve_ivp(fun=rhs, t_span=[tvals[0], tvals[-1]], y0=[0,0], t_eval=tvals, args=(L, Fmagx, Fmagy, mu, a, h, U0,))

#sol.y[0] is x pos at each of the tvals points, sol.y[1] is y pos at each of the tvals points
xsol = sol.y[0]
ysol = sol.y[1]


plt.subplot(2,1,1)
plt.plot(tvals, xsol)
plt.ylabel('x')
plt.subplot(2,1,2)
plt.plot(tvals, ysol)
plt.ylabel('y')
plt.xlabel('t')
plt.show()

plt.plot(xsol, ysol)
# plt.plot(sol.y[0], sol.y[1])
#Plotting start and end points to see what's going on 
plt.plot(sol.y[0,0],sol.y[1,0], marker="o", color='red')
plt.plot(sol.y[0,50],sol.y[1,50], marker="o", color='orange')
plt.plot(sol.y[0,-1],sol.y[1,-1], marker="o", color='blue')
# print(sol.y_events)
plt.show()


# plt.plot(xsol, ysol)
# # plt.plot(sol.y[0], sol.y[1])
# #Plotting start and end points to see what's going on 
# plt.plot(sol.y[0,0],sol.y[1,0], marker="o", color='red')
# plt.plot(sol.y[0,50],sol.y[1,50], marker="o", color='orange')
# plt.plot(sol.y[0,-1],sol.y[1,-1], marker="o", color='blue')
# # print(sol.y_events)
# plt.ylim(0,0.02)
# plt.show()