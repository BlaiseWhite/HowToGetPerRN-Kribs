import math
import numpy as np
import pandas as pd
import scipy
from scipy import integrate
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt

p1 = 0.05
p2 = .1
tau = 0.5
omega = 1
mu_h = 1
mu_v = 1.562
beta_v1 = 1.59
beta_v2 = 1.59
a1 = 1.2
a2 = 0.8
rho1 = 0.177
rho2 = 0.177
Nh = 9.3
Nv = 128
Qh = 10
Qv = 100
H = 1

beta_h1 = 0.823
beta_h2 = 0.823

lamb = 1
lamb2 = 1
lamb11 = 1
lamb12 = 1

i= 1
x01 = np.array([1, 0])
x02 = np.array([0, 1])
t = np.linspace(0, 1, 365)

#Functions 1

def beta_h_func(beta_h):
    return beta_h1*min(Nv/(Qh*Nh), 1) + rho1*H*min(Nv/(Qh*Nh), 1)

def beta_v_func(t):
    p = a1 if t < i/2 else (omega - a1*tau)/(omega-tau)
    return beta_v1*min((Qv*Nh)/Nv, 1)*p

def ode1(t, x):
    x1, x2 = x
    dx1dt = ((p1/lamb) - 1)*mu_h*x1 + (((beta_h_func(beta_h1))/lamb)*(Nh/Nv))*x2
    dx2dt = (((beta_v_func(t))/lamb)*(Nv/Nh))*x1 - mu_v*x2
    return [dx1dt, dx2dt]

def ode2(t, x):
    x1, x2 = x
    dx3dt = ((p1/lamb11)-1)*mu_h*x1+((beta_h_func(beta_h1)/lamb11)*(Sh(t,x1)/Nv)*x2)
    dx4dt = ((beta_v_func(t)/lamb11)*(Sv(t,x2)/Nh)*x1)-mu_v*x2
    return [dx3dt,dx4dt]

def ode3(t, x):
    x1, x2 = x
    dx5dt = beta_h2_func(beta_h2)*(Nh-x1)*(x2/Nv)-mu_h*(1-p1)*x1
    dx6dt = beta_v2_func(t)*(Nv-x2)*(x1/Nh)-mu_v*x2
    return [dx5dt,dx6dt]

def Sh(t, x):
    x1 = x
    Sha = Nh - x1
    return Sha

def Sv(t, x):
    x2 = x
    Sva = Nv - x2
    return Sva

#Functions 2

def beta_h2_func(beta_h):
    return beta_h2*min(Nv/(Qh*Nh), 1) + rho2*H*min(Nv/(Qh*Nh), 1)

def beta_v2_func(t):
    p = a2 if t < i/2 else (omega - a2*tau)/(omega-tau)
    return beta_v2*min((Qv*Nh)/Nv, 1)*p

def ode12(t, x):
    x1, x2 = x
    dx7dt = ((p2/lamb2) - 1)*mu_h*x1 + (((beta_h2_func(beta_h2))/lamb2)*(Nh/Nv))*x2
    dx8dt = (((beta_v2_func(t))/lamb2)*(Nv/Nh))*x1 - mu_v*x2
    return [dx7dt, dx8dt]

def ode22(t, x):
    x1, x2 = x
    dx9dt = ((p2/lamb12)-1)*mu_h*x1+((beta_h2_func(beta_h2)/lamb12)*(Sh2(t,x1)/Nv)*x2)
    dx10dt = ((beta_v2_func(t)/lamb12)*(Sv2(t,x2)/Nh)*x1)-mu_v*x2
    return [dx9dt,dx10dt]


def ode32(t, x):
    x1, x2 = x
    dx11dt = beta_h_func(beta_h1)*(Nh-x1)*(x2/Nv)-mu_h*(1-p2)*x1
    dx12dt = beta_v_func(t)*(Nv-x2)*(x1/Nh)-mu_v*x2
    return [dx11dt,dx12dt]
        

def Sh2(t, x):
    x1 = x
    Sha2 = Nh - x1
    return Sha2

def Sv2(t, x):
    x2 = x
    Sva2 = Nv - x2
    return Sva2

#BRN1
sol1 = solve_ivp(ode1, [t.min(), t.max()], x01, t_eval=t)
v1 = sol1.y[:, -2:-1]
sol2 = solve_ivp(ode1, [t.min(), t.max()], x02, t_eval=t)
v2 = sol2.y[:, -2:-1]
rho11 = np.max(np.linalg.eigvals(np.hstack((v1, v2))))
BRN1 = lamb

while rho11 > 1:
    lamb += .0001
    sol1 = solve_ivp(ode1, [t.min(), t.max()], x01, t_eval=t)
    v1 = sol1.y[:, -2:-1]
    sol2 = solve_ivp(ode1, [t.min(), t.max()], x02, t_eval=t)
    v2 = sol2.y[:, -2:-1]
    rho11 = np.max(np.linalg.eigvals(np.hstack((v1, v2))))
    BRN1 = lamb
    
while rho11 < 1:
    lamb -= .0001
    sol1 = solve_ivp(ode1, [t.min(), t.max()], x01, t_eval=t)
    v1 = sol1.y[:, -2:-1]
    sol2 = solve_ivp(ode1, [t.min(), t.max()], x02, t_eval=t)
    v2 = sol2.y[:, -2:-1]
    rho11 = np.max(np.linalg.eigvals(np.hstack((v1, v2))))
    BRN1 = lamb
    
if rho11 == 1:
    BRN1 = lamb

#BRN2

sol12 = solve_ivp(ode12, [t.min(), t.max()], x01, t_eval=t)
v12 = sol12.y[:, -2:-1]
sol22 = solve_ivp(ode12, [t.min(), t.max()], x02, t_eval=t)
v22 = sol22.y[:, -2:-1]
rho112 = np.max(np.linalg.eigvals(np.hstack((v12, v22))))
BRN2 = lamb2

while rho112 > 1:
    lamb2 += .0001
    sol12 = solve_ivp(ode12, [t.min(), t.max()], x01, t_eval=t)
    v12 = sol12.y[:, -2:-1]
    sol22 = solve_ivp(ode12, [t.min(), t.max()], x02, t_eval=t)
    v22 = sol22.y[:, -2:-1]
    rho112 = np.max(np.linalg.eigvals(np.hstack((v12, v22))))
    BRN2 = lamb2
    
while rho112 < 1:
    lamb2 -= .0001
    sol12 = solve_ivp(ode12, [t.min(), t.max()], x01, t_eval=t)
    v12 = sol12.y[:, -2:-1]
    sol22 = solve_ivp(ode12, [t.min(), t.max()], x02, t_eval=t)
    v22 = sol22.y[:, -2:-1]
    rho112 = np.max(np.linalg.eigvals(np.hstack((v12, v22))))
    BRN2 = lamb2
    
if rho112 == 1:
    BRN2 = lamb2

#IRN(Bhi,Bhj) Bhj = Bh(3-i) ie Bhi(1) = Bhj(2)
resultIRN1BRN1 = []
resultIRN1BRN2 = []
resultIRN1BRN1.append(BRN1)
resultIRN1BRN2.append(BRN2)
x1list = []
x2list = []
tlist = []
while i < 10:
    i += 1
    t = np.linspace(i, 1+i, 365*i)
    
    def beta_v2_func(t):
        p = a2 if t < i/2 else (omega - a2*tau)/(omega-tau)
        return beta_v2*min((Qv*Nh)/Nv, 1)*p
    
    def ode3(t, x):
        x1, x2 = x
        dx5dt = beta_h2_func(beta_h2)*(Nh-x1)*(x2/Nv)-mu_h*(1-p1)*x1
        dx6dt = beta_v2_func(t)*(Nv-x2)*(x1/Nh)-mu_v*x2
        return [dx5dt,dx6dt]
    
    sol5 = solve_ivp(ode3, [t.min(), t.max()], x01, t_eval=t)
    v5 = sol5.y[: , -2:-1]
    sol6 = solve_ivp(ode3, [t.min(), t.max()], x02, t_eval=t)
    v6 = sol6.y[: , -2:-1]
    rho13 = np.max(np.linalg.eigvals(np.hstack((v5, v6))))
    
    x1 = sol5.y[0]
    x2 = sol5.y[1]
    
    x1list.extend(x1)
    tlist.extend(t)
    x2list.extend(x2)

t = np.linspace(0, 1*i, 365*i)

#2w is the period of oscillation
x1 = x1list[0:730]
x2 = x2list[0:730]


sol7 = solve_ivp(ode2, [t.min(), t.max()], x01, t_eval=t)
v7 = sol7.y[: , -2:-1]
sol8 = solve_ivp(ode2, [t.min(), t.max()], x02, t_eval=t)
v8 = sol8.y[: , -2:-1]
rho14 = np.max(np.linalg.eigvals(np.hstack((v7, v8))))
IRN1 = lamb11

while rho14 < 1:
    lamb11 -= .0001
    sol7 = solve_ivp(ode2, [t.min(), t.max()], x01, t_eval=t)
    v7 = sol7.y[: , -2:-1]
    sol8 = solve_ivp(ode2, [t.min(), t.max()], x02, t_eval=t)
    v8 = sol8.y[: , -2:-1]
    rho14 = np.max(np.linalg.eigvals(np.hstack((v7, v8))))
    IRN1 = lamb11    

while rho14 > 1:
    lamb11 += .0001
    sol7 = solve_ivp(ode2, [t.min(), t.max()], x01, t_eval=t)
    v7 = sol7.y[: , -2:-1]
    sol8 = solve_ivp(ode2, [t.min(), t.max()], x02, t_eval=t)
    v8 = sol8.y[: , -2:-1]
    rho14 = np.max(np.linalg.eigvals(np.hstack((v7, v8))))
    IRN1 = lamb11

if rho14 == 1:
    IRN1 = lamb11

##2
#IRN(Bhi,Bhj) Bhj = Bh(3-i) ie Bhi(1) = Bhj(2)

sol52 = solve_ivp(ode32, [t.min(), t.max()], x01, t_eval=t)
v52 = sol52.y[: , -2:-1]
sol62 = solve_ivp(ode32, [t.min(), t.max()], x02, t_eval=t)
v62 = sol62.y[: , -2:-1]
rho132 = np.max(np.linalg.eigvals(np.hstack((v52, v62))))

sol72 = solve_ivp(ode22, [t.min(), t.max()], x01, t_eval=t)
v72 = sol72.y[: , -2:-1]
sol82 = solve_ivp(ode22, [t.min(), t.max()], x02, t_eval=t)
v82 = sol82.y[: , -2:-1]
rho142 = np.max(np.linalg.eigvals(np.hstack((v72, v82))))
IRN2 = lamb12

while rho142 > 1:
    lamb12 += .0001
    sol72 = solve_ivp(ode22, [t.min(), t.max()], x01, t_eval=t)
    v72 = sol72.y[: , -2:-1]
    sol82 = solve_ivp(ode22, [t.min(), t.max()], x02, t_eval=t)
    v82 = sol82.y[: , -2:-1]
    rho142 = np.max(np.linalg.eigvals(np.hstack((v72, v82))))
    IRN2 = lamb12

while rho142 < 1:
    lamb12 -= .0001
    sol72 = solve_ivp(ode22, [t.min(), t.max()], x01, t_eval=t)
    v72 = sol72.y[: , -2:-1]
    sol82 = solve_ivp(ode22, [t.min(), t.max()], x02, t_eval=t)
    v82 = sol82.y[: , -2:-1]
    rho142 = np.max(np.linalg.eigvals(np.hstack((v72, v82))))
    IRN2 = lamb12

if rho142 == 1:
    IRN2 = lamb12

#Graph

print("BRN(beta_h1) is:", BRN1)
print("BRN(beta_h2)", BRN2)
print("The value of betah1 is:", beta_h1)
print("The value of betah2 is:", beta_h2)
print("The value of IRN1 is:", IRN1)
print("The value of IRN2 is:", IRN2)

IRN1list = []

i = 1
while i < 100:
    i += 1
    beta_h2 = 0.8229999999999239+(00.8229999999999239*i*.1)

    #IRN(Bhi,Bhj) Bhj = Bh(3-i) ie Bhi(1) = Bhj(2)

    sol5 = solve_ivp(ode3, [t.min(), t.max()], x01, t_eval=t)
    v5 = sol5.y[: , -2:-1]
    sol6 = solve_ivp(ode3, [t.min(), t.max()], x02, t_eval=t)
    v6 = sol6.y[: , -2:-1]
    rho13 = np.max(np.linalg.eigvals(np.hstack((v5, v6))))

    sol7 = solve_ivp(ode2, [t.min(), t.max()], x01, t_eval=t)
    v7 = sol7.y[: , -2:-1]
    sol8 = solve_ivp(ode2, [t.min(), t.max()], x02, t_eval=t)
    v8 = sol8.y[: , -2:-1]
    rho14 = np.max(np.linalg.eigvals(np.hstack((v7, v8))))
    IRN1 = lamb11
    
    t= np.linspace(0, i, 365*i)
    
    while IRN1 < 1:
        beta_h1 += .0001
            
        sol5 = solve_ivp(ode3, [t.min(), t.max()], x01, t_eval=t)
        v5 = sol5.y[: , -2:-1]
        sol6 = solve_ivp(ode3, [t.min(), t.max()], x02, t_eval=t)
        v6 = sol6.y[: , -2:-1]
        rho13 = np.max(np.linalg.eigvals(np.hstack((v5, v6))))

        sol7 = solve_ivp(ode2, [t.min(), t.max()], x01, t_eval=t)
        v7 = sol7.y[: , -2:-1]
        sol8 = solve_ivp(ode2, [t.min(), t.max()], x02, t_eval=t)
        v8 = sol8.y[: , -2:-1]
        rho14 = np.max(np.linalg.eigvals(np.hstack((v7, v8))))
        IRN1 = lamb11
            
        while rho14 < 1:
            lamb11 -= .0001
            sol5 = solve_ivp(ode3, [t.min(), t.max()], x01, t_eval=t)
            v5 = sol5.y[: , -2:-1]
            sol6 = solve_ivp(ode3, [t.min(), t.max()], x02, t_eval=t)
            v6 = sol6.y[: , -2:-1]
            rho13 = np.max(np.linalg.eigvals(np.hstack((v5, v6))))

            sol7 = solve_ivp(ode2, [t.min(), t.max()], x01, t_eval=t)
            v7 = sol7.y[: , -2:-1]
            sol8 = solve_ivp(ode2, [t.min(), t.max()], x02, t_eval=t)
            v8 = sol8.y[: , -2:-1]
            rho14 = np.max(np.linalg.eigvals(np.hstack((v7, v8))))
            IRN1 = lamb11
            if rho14 == 1:
                IRN1 = lamb11
                    
        while rho14 > 1:
            lamb11 += .0001
            sol5 = solve_ivp(ode3, [t.min(), t.max()], x01, t_eval=t)
            v5 = sol5.y[: , -2:-1]
            sol6 = solve_ivp(ode3, [t.min(), t.max()], x02, t_eval=t)
            v6 = sol6.y[: , -2:-1]
            rho13 = np.max(np.linalg.eigvals(np.hstack((v5, v6))))

            sol7 = solve_ivp(ode2, [t.min(), t.max()], x01, t_eval=t)
            v7 = sol7.y[: , -2:-1]
            sol8 = solve_ivp(ode2, [t.min(), t.max()], x02, t_eval=t)
            v8 = sol8.y[: , -2:-1]
            rho14 = np.max(np.linalg.eigvals(np.hstack((v7, v8))))
            IRN1 = lamb11
            if rho14 == 1:
                IRN1 = lamb11

    if IRN1 == 1:
        IRN1list.append(IRN1)
        sol1 = solve_ivp(ode1, [t.min(), t.max()], x01, t_eval=t)
        v1 = sol1.y[:, -2:-1]
        sol2 = solve_ivp(ode1, [t.min(), t.max()], x02, t_eval=t)
        v2 = sol2.y[:, -2:-1]
        rho11 = np.max(np.linalg.eigvals(np.hstack((v1, v2))))
        BRN1 = lamb

        while rho11 > 1:
            lamb += .0001
            sol1 = solve_ivp(ode1, [t.min(), t.max()], x01, t_eval=t)
            v1 = sol1.y[:, -2:-1]
            sol2 = solve_ivp(ode1, [t.min(), t.max()], x02, t_eval=t)
            v2 = sol2.y[:, -2:-1]
            rho11 = np.max(np.linalg.eigvals(np.hstack((v1, v2))))
            BRN1 = lamb
    
        while rho11 < 1:
            lamb -= .0001
            sol1 = solve_ivp(ode1, [t.min(), t.max()], x01, t_eval=t)
            v1 = sol1.y[:, -2:-1]
            sol2 = solve_ivp(ode1, [t.min(), t.max()], x02, t_eval=t)
            v2 = sol2.y[:, -2:-1]
            rho11 = np.max(np.linalg.eigvals(np.hstack((v1, v2))))
            BRN1 = lamb
    
        if rho11 == 1:
            BRN1 = lamb
                
        sol12 = solve_ivp(ode12, [t.min(), t.max()], x01, t_eval=t)
        v12 = sol12.y[:, -2:-1]
        sol22 = solve_ivp(ode12, [t.min(), t.max()], x02, t_eval=t)
        v22 = sol22.y[:, -2:-1]
        rho112 = np.max(np.linalg.eigvals(np.hstack((v12, v22))))
        BRN2 = lamb2

        while rho112 > 1:
            lamb2 += .0001
            sol12 = solve_ivp(ode12, [t.min(), t.max()], x01, t_eval=t)
            v12 = sol12.y[:, -2:-1]
            sol22 = solve_ivp(ode12, [t.min(), t.max()], x02, t_eval=t)
            v22 = sol22.y[:, -2:-1]
            rho112 = np.max(np.linalg.eigvals(np.hstack((v12, v22))))
            BRN2 = lamb2
    
        while rho112 < 1:
            lamb2 -= .0001
            sol12 = solve_ivp(ode12, [t.min(), t.max()], x01, t_eval=t)
            v12 = sol12.y[:, -2:-1]
            sol22 = solve_ivp(ode12, [t.min(), t.max()], x02, t_eval=t)
            v22 = sol22.y[:, -2:-1]
            rho112 = np.max(np.linalg.eigvals(np.hstack((v12, v22))))
            BRN2 = lamb2
    
        if rho112 == 1:
            BRN2 = lamb2
                

    while IRN1 > 1:
        beta_h1 -= .0001
            
        sol5 = solve_ivp(ode3, [t.min(), t.max()], x01, t_eval=t)
        v5 = sol5.y[: , -2:-1]
        sol6 = solve_ivp(ode3, [t.min(), t.max()], x02, t_eval=t)
        v6 = sol6.y[: , -2:-1]
        rho13 = np.max(np.linalg.eigvals(np.hstack((v5, v6))))

        sol7 = solve_ivp(ode2, [t.min(), t.max()], x01, t_eval=t)
        v7 = sol7.y[: , -2:-1]
        sol8 = solve_ivp(ode2, [t.min(), t.max()], x02, t_eval=t)
        v8 = sol8.y[: , -2:-1]
        rho14 = np.max(np.linalg.eigvals(np.hstack((v7, v8))))
        IRN1 = lamb11
            
        while rho14 < 1:
            lamb11 -= .0001
            sol5 = solve_ivp(ode3, [t.min(), t.max()], x01, t_eval=t)
            v5 = sol5.y[: , -2:-1]
            sol6 = solve_ivp(ode3, [t.min(), t.max()], x02, t_eval=t)
            v6 = sol6.y[: , -2:-1]
            rho13 = np.max(np.linalg.eigvals(np.hstack((v5, v6))))

            sol7 = solve_ivp(ode2, [t.min(), t.max()], x01, t_eval=t)
            v7 = sol7.y[: , -2:-1]
            sol8 = solve_ivp(ode2, [t.min(), t.max()], x02, t_eval=t)
            v8 = sol8.y[: , -2:-1]
            rho14 = np.max(np.linalg.eigvals(np.hstack((v7, v8))))
            
            if rho14 == 1:
                IRN1 = lamb11
                    
        while rho14 > 1:
            lamb11 += .0001
            sol5 = solve_ivp(ode3, [t.min(), t.max()], x01, t_eval=t)
            v5 = sol5.y[: , -2:-1]
            sol6 = solve_ivp(ode3, [t.min(), t.max()], x02, t_eval=t)
            v6 = sol6.y[: , -2:-1]
            rho13 = np.max(np.linalg.eigvals(np.hstack((v5, v6))))

            sol7 = solve_ivp(ode2, [t.min(), t.max()], x01, t_eval=t)
            v7 = sol7.y[: , -2:-1]
            sol8 = solve_ivp(ode2, [t.min(), t.max()], x02, t_eval=t)
            v8 = sol8.y[: , -2:-1]
            rho14 = np.max(np.linalg.eigvals(np.hstack((v7, v8))))
            if rho14 == 1:
                IRN1 = lamb11

    if IRN1 == 1:
        IRN1list.append(IRN1)
        sol1 = solve_ivp(ode1, [t.min(), t.max()], x01, t_eval=t)
        v1 = sol1.y[:, -2:-1]
        sol2 = solve_ivp(ode1, [t.min(), t.max()], x02, t_eval=t)
        v2 = sol2.y[:, -2:-1]
        rho11 = np.max(np.linalg.eigvals(np.hstack((v1, v2))))
        BRN1 = lamb

        while rho11 > 1:
            lamb += .0001
            sol1 = solve_ivp(ode1, [t.min(), t.max()], x01, t_eval=t)
            v1 = sol1.y[:, -2:-1]
            sol2 = solve_ivp(ode1, [t.min(), t.max()], x02, t_eval=t)
            v2 = sol2.y[:, -2:-1]
            rho11 = np.max(np.linalg.eigvals(np.hstack((v1, v2))))
            BRN1 = lamb
    
        while rho11 < 1:
            lamb -= .0001
            sol1 = solve_ivp(ode1, [t.min(), t.max()], x01, t_eval=t)
            v1 = sol1.y[:, -2:-1]
            sol2 = solve_ivp(ode1, [t.min(), t.max()], x02, t_eval=t)
            v2 = sol2.y[:, -2:-1]
            rho11 = np.max(np.linalg.eigvals(np.hstack((v1, v2))))
            BRN1 = lamb
    
        if rho11 == 1:
            BRN1 = lamb
                
        sol12 = solve_ivp(ode12, [t.min(), t.max()], x01, t_eval=t)
        v12 = sol12.y[:, -2:-1]
        sol22 = solve_ivp(ode12, [t.min(), t.max()], x02, t_eval=t)
        v22 = sol22.y[:, -2:-1]
        rho112 = np.max(np.linalg.eigvals(np.hstack((v12, v22))))
        BRN2 = lamb2

        while rho112 > 1:
            lamb2 += .0001
            sol12 = solve_ivp(ode12, [t.min(), t.max()], x01, t_eval=t)
            v12 = sol12.y[:, -2:-1]
            sol22 = solve_ivp(ode12, [t.min(), t.max()], x02, t_eval=t)
            v22 = sol22.y[:, -2:-1]
            rho112 = np.max(np.linalg.eigvals(np.hstack((v12, v22))))
            BRN2 = lamb2
    
        while rho112 < 1:
            lamb2 -= .0001
            sol12 = solve_ivp(ode12, [t.min(), t.max()], x01, t_eval=t)
            v12 = sol12.y[:, -2:-1]
            sol22 = solve_ivp(ode12, [t.min(), t.max()], x02, t_eval=t)
            v22 = sol22.y[:, -2:-1]
            rho112 = np.max(np.linalg.eigvals(np.hstack((v12, v22))))
            BRN2 = lamb2
    
        if rho112 == 1:
            BRN2 = lamb2
            
    resultIRN1BRN1.append(BRN1)
    resultIRN1BRN2.append(BRN2)

plt.plot(resultIRN1BRN1, resultIRN1BRN2, label="IRN1")
plt.xlabel("BRN(beta_h1)")
plt.ylabel("BRN(beta_h2)")
plt.legend(loc='upper right')
plt.title("IRN1 and IRN2")


###IRN2

beta_h1 = 0.823
beta_h2 = 0.823

lamb = 1
lamb2 = 1
lamb11 = 1
lamb12 = 1

i= 1
x01 = np.array([1, 0])
x02 = np.array([0, 1])
t = np.linspace(0, 1, 365)

#Functions 1

def beta_h_func(beta_h):
    return beta_h1*min(Nv/(Qh*Nh), 1) + rho1*H*min(Nv/(Qh*Nh), 1)

def beta_v_func(t):
    p = a1 if t < i/2 else (omega - a1*tau)/(omega-tau)
    return beta_v1*min((Qv*Nh)/Nv, 1)*p

def ode1(t, x):
    x1, x2 = x
    dx1dt = ((p1/lamb) - 1)*mu_h*x1 + (((beta_h_func(beta_h1))/lamb)*(Nh/Nv))*x2
    dx2dt = (((beta_v_func(t))/lamb)*(Nv/Nh))*x1 - mu_v*x2
    return [dx1dt, dx2dt]

def ode2(t, x):
    x1, x2 = x
    dx3dt = ((p1/lamb11)-1)*mu_h*x1+((beta_h_func(beta_h1)/lamb11)*(Sh(t,x1)/Nv)*x2)
    dx4dt = ((beta_v_func(t)/lamb11)*(Sv(t,x2)/Nh)*x1)-mu_v*x2
    return [dx3dt,dx4dt]

def ode3(t, x):
    x1, x2 = x
    dx5dt = beta_h2_func(beta_h2)*(Nh-x1)*(x2/Nv)-mu_h*(1-p1)*x1
    dx6dt = beta_v2_func(t)*(Nv-x2)*(x1/Nh)-mu_v*x2
    return [dx5dt,dx6dt]

def Sh(t, x):
    x1 = x
    Sha = Nh - x1
    return Sha

def Sv(t, x):
    x2 = x
    Sva = Nv - x2
    return Sva

#Functions 2

def beta_h2_func(beta_h):
    return beta_h2*min(Nv/(Qh*Nh), 1) + rho2*H*min(Nv/(Qh*Nh), 1)

def beta_v2_func(t):
    p = a2 if t < i/2 else (omega - a2*tau)/(omega-tau)
    return beta_v2*min((Qv*Nh)/Nv, 1)*p

def ode12(t, x):
    x1, x2 = x
    dx7dt = ((p2/lamb2) - 1)*mu_h*x1 + (((beta_h2_func(beta_h2))/lamb2)*(Nh/Nv))*x2
    dx8dt = (((beta_v2_func(t))/lamb2)*(Nv/Nh))*x1 - mu_v*x2
    return [dx7dt, dx8dt]

def ode22(t, x):
    x1, x2 = x
    dx9dt = ((p2/lamb12)-1)*mu_h*x1+((beta_h2_func(beta_h2)/lamb12)*(Sh2(t,x1)/Nv)*x2)
    dx10dt = ((beta_v2_func(t)/lamb12)*(Sv2(t,x2)/Nh)*x1)-mu_v*x2
    return [dx9dt,dx10dt]


def ode32(t, x):
    x1, x2 = x
    dx11dt = beta_h_func(beta_h1)*(Nh-x1)*(x2/Nv)-mu_h*(1-p2)*x1
    dx12dt = beta_v_func(t)*(Nv-x2)*(x1/Nh)-mu_v*x2
    return [dx11dt,dx12dt]
        

def Sh2(t, x):
    x1 = x
    Sha2 = Nh - x1
    return Sha2

def Sv2(t, x):
    x2 = x
    Sva2 = Nv - x2
    return Sva2


#BRN1
sol1 = solve_ivp(ode1, [t.min(), t.max()], x01, t_eval=t)
v1 = sol1.y[:, -2:-1]
sol2 = solve_ivp(ode1, [t.min(), t.max()], x02, t_eval=t)
v2 = sol2.y[:, -2:-1]
rho11 = np.max(np.linalg.eigvals(np.hstack((v1, v2))))
BRN1 = lamb

while rho11 > 1:
    lamb += .0001
    sol1 = solve_ivp(ode1, [t.min(), t.max()], x01, t_eval=t)
    v1 = sol1.y[:, -2:-1]
    sol2 = solve_ivp(ode1, [t.min(), t.max()], x02, t_eval=t)
    v2 = sol2.y[:, -2:-1]
    rho11 = np.max(np.linalg.eigvals(np.hstack((v1, v2))))
    BRN1 = lamb
    
while rho11 < 1:
    lamb -= .0001
    sol1 = solve_ivp(ode1, [t.min(), t.max()], x01, t_eval=t)
    v1 = sol1.y[:, -2:-1]
    sol2 = solve_ivp(ode1, [t.min(), t.max()], x02, t_eval=t)
    v2 = sol2.y[:, -2:-1]
    rho11 = np.max(np.linalg.eigvals(np.hstack((v1, v2))))
    BRN1 = lamb
    
if rho11 == 1:
    BRN1 = lamb

#BRN2

sol12 = solve_ivp(ode12, [t.min(), t.max()], x01, t_eval=t)
v12 = sol12.y[:, -2:-1]
sol22 = solve_ivp(ode12, [t.min(), t.max()], x02, t_eval=t)
v22 = sol22.y[:, -2:-1]
rho112 = np.max(np.linalg.eigvals(np.hstack((v12, v22))))
BRN2 = lamb2

while rho112 > 1:
    lamb2 += .0001
    sol12 = solve_ivp(ode12, [t.min(), t.max()], x01, t_eval=t)
    v12 = sol12.y[:, -2:-1]
    sol22 = solve_ivp(ode12, [t.min(), t.max()], x02, t_eval=t)
    v22 = sol22.y[:, -2:-1]
    rho112 = np.max(np.linalg.eigvals(np.hstack((v12, v22))))
    BRN2 = lamb2
    
while rho112 < 1:
    lamb2 -= .0001
    sol12 = solve_ivp(ode12, [t.min(), t.max()], x01, t_eval=t)
    v12 = sol12.y[:, -2:-1]
    sol22 = solve_ivp(ode12, [t.min(), t.max()], x02, t_eval=t)
    v22 = sol22.y[:, -2:-1]
    rho112 = np.max(np.linalg.eigvals(np.hstack((v12, v22))))
    BRN2 = lamb2
    
if rho112 == 1:
    BRN2 = lamb2

#IRN2(Bhi,Bhj) Bhj = Bh(3-i) ie Bhi(1) = Bhj(2)
resultIRN2BRN1 = []
resultIRN2BRN2 = []
resultIRN2BRN1.append(BRN1)
resultIRN2BRN2.append(BRN2)
x1list = []
x2list = []
tlist = []

while i < 10:
    i += 1
    t = np.linspace(i, 1+i, 365*i)
    
    def beta_v2_func(t):
        p = a2 if t < i/2 else (omega - a2*tau)/(omega-tau)
        return beta_v2*min((Qv*Nh)/Nv, 1)*p
    
    def ode3(t, x):
        x1, x2 = x
        dx5dt = beta_h2_func(beta_h2)*(Nh-x1)*(x2/Nv)-mu_h*(1-p1)*x1
        dx6dt = beta_v2_func(t)*(Nv-x2)*(x1/Nh)-mu_v*x2
        return [dx5dt,dx6dt]
    
    sol5 = solve_ivp(ode3, [t.min(), t.max()], x01, t_eval=t)
    v5 = sol5.y[: , -2:-1]
    sol6 = solve_ivp(ode3, [t.min(), t.max()], x02, t_eval=t)
    v6 = sol6.y[: , -2:-1]
    rho13 = np.max(np.linalg.eigvals(np.hstack((v5, v6))))
    
    sol3 = solve_ivp(ode3, [t.min(), t.max()], x01, t_eval=t)
    v3 = sol3.y[:, -2:-1]
    sol4 = solve_ivp(ode3, [t.min(), t.max()], x02, t_eval=t)
    v4 = sol4.y[:, -2:-1]
    rho15 = np.max(np.linalg.eigvals(np.hstack((v3,v4))))

    x1 = sol5.y[0]
    x2 = sol5.y[1]

    
    x1list.extend(x1)
    tlist.extend(t)
    x2list.extend(x2)

t = np.linspace(0, 1*i, 365*i)

#2w is the period of oscillation
x1 = x1list[0:730]
x2 = x2list[0:730]


sol7 = solve_ivp(ode2, [t.min(), t.max()], x01, t_eval=t)
v7 = sol7.y[: , -2:-1]
sol8 = solve_ivp(ode2, [t.min(), t.max()], x02, t_eval=t)
v8 = sol8.y[: , -2:-1]
rho14 = np.max(np.linalg.eigvals(np.hstack((v7, v8))))
IRN1 = lamb11

while rho14 < 1:
    lamb11 -= .0001
    sol7 = solve_ivp(ode2, [t.min(), t.max()], x01, t_eval=t)
    v7 = sol7.y[: , -2:-1]
    sol8 = solve_ivp(ode2, [t.min(), t.max()], x02, t_eval=t)
    v8 = sol8.y[: , -2:-1]
    rho14 = np.max(np.linalg.eigvals(np.hstack((v7, v8))))
    IRN1 = lamb11    

while rho14 > 1:
    lamb11 += .0001
    sol7 = solve_ivp(ode2, [t.min(), t.max()], x01, t_eval=t)
    v7 = sol7.y[: , -2:-1]
    sol8 = solve_ivp(ode2, [t.min(), t.max()], x02, t_eval=t)
    v8 = sol8.y[: , -2:-1]
    rho14 = np.max(np.linalg.eigvals(np.hstack((v7, v8))))
    IRN1 = lamb11

if rho14 == 1:
    IRN1 = lamb11

##2
#IRN(Bhi,Bhj) Bhj = Bh(3-i) ie Bhi(1) = Bhj(2)

sol52 = solve_ivp(ode32, [t.min(), t.max()], x01, t_eval=t)
v52 = sol52.y[: , -2:-1]
sol62 = solve_ivp(ode32, [t.min(), t.max()], x02, t_eval=t)
v62 = sol62.y[: , -2:-1]
rho132 = np.max(np.linalg.eigvals(np.hstack((v52, v62))))

sol72 = solve_ivp(ode22, [t.min(), t.max()], x01, t_eval=t)
v72 = sol72.y[: , -2:-1]
sol82 = solve_ivp(ode22, [t.min(), t.max()], x02, t_eval=t)
v82 = sol82.y[: , -2:-1]
rho142 = np.max(np.linalg.eigvals(np.hstack((v72, v82))))
IRN2 = lamb12

while rho142 > 1:
    lamb12 += .0001
    sol72 = solve_ivp(ode22, [t.min(), t.max()], x01, t_eval=t)
    v72 = sol72.y[: , -2:-1]
    sol82 = solve_ivp(ode22, [t.min(), t.max()], x02, t_eval=t)
    v82 = sol82.y[: , -2:-1]
    rho142 = np.max(np.linalg.eigvals(np.hstack((v72, v82))))
    IRN2 = lamb12

while rho142 < 1:
    lamb12 -= .0001
    sol72 = solve_ivp(ode22, [t.min(), t.max()], x01, t_eval=t)
    v72 = sol72.y[: , -2:-1]
    sol82 = solve_ivp(ode22, [t.min(), t.max()], x02, t_eval=t)
    v82 = sol82.y[: , -2:-1]
    rho142 = np.max(np.linalg.eigvals(np.hstack((v72, v82))))
    IRN2 = lamb12

if rho142 == 1:
    IRN2 = lamb12

#Graph

print("BRN(beta_h1) is:", BRN1)
print("BRN(beta_h2)", BRN2)
print("The value of betah1 is:", beta_h1)
print("The value of betah2 is:", beta_h2)
print("The value of IRN1 is:", IRN1)
print("The value of IRN2 is:", IRN2)

IRN2list = []

i = 0
while i < 100:
    i += 1
    beta_h2 = 0.8229999999999239+(00.8229999999999239*i*.1)

    #IRN(Bhi,Bhj) Bhj = Bh(3-i) ie Bhi(1) = Bhj(2)

    sol52 = solve_ivp(ode32, [t.min(), t.max()], x01, t_eval=t)
    v52 = sol52.y[: , -2:-1]
    sol62 = solve_ivp(ode32, [t.min(), t.max()], x02, t_eval=t)
    v62 = sol62.y[: , -2:-1]
    rho132 = np.max(np.linalg.eigvals(np.hstack((v52, v62))))

    sol72 = solve_ivp(ode22, [t.min(), t.max()], x01, t_eval=t)
    v72 = sol72.y[: , -2:-1]
    sol82 = solve_ivp(ode22, [t.min(), t.max()], x02, t_eval=t)
    v82 = sol82.y[: , -2:-1]
    rho142 = np.max(np.linalg.eigvals(np.hstack((v72, v82))))
    IRN2 = lamb12
    
    t= np.linspace(0, i, 365*i)
    
    while IRN2 < 1:
        beta_h2 += .0001
            
        sol52 = solve_ivp(ode32, [t.min(), t.max()], x01, t_eval=t)
        v52 = sol52.y[: , -2:-1]
        sol62 = solve_ivp(ode32, [t.min(), t.max()], x02, t_eval=t)
        v62 = sol62.y[: , -2:-1]
        rho132 = np.max(np.linalg.eigvals(np.hstack((v52, v62))))

        sol72 = solve_ivp(ode22, [t.min(), t.max()], x01, t_eval=t)
        v72 = sol72.y[: , -2:-1]
        sol82 = solve_ivp(ode22, [t.min(), t.max()], x02, t_eval=t)
        v82 = sol82.y[: , -2:-1]
        rho142 = np.max(np.linalg.eigvals(np.hstack((v72, v82))))
        IRN2 = lamb12
            
        while rho142 < 1:
            lamb12 -= .0001
            sol52 = solve_ivp(ode32, [t.min(), t.max()], x01, t_eval=t)
            v52 = sol52.y[: , -2:-1]
            sol62 = solve_ivp(ode32, [t.min(), t.max()], x02, t_eval=t)
            v62 = sol62.y[: , -2:-1]
            rho132 = np.max(np.linalg.eigvals(np.hstack((v52, v62))))

            sol72 = solve_ivp(ode22, [t.min(), t.max()], x01, t_eval=t)
            v72 = sol72.y[: , -2:-1]
            sol82 = solve_ivp(ode22, [t.min(), t.max()], x02, t_eval=t)
            v82 = sol82.y[: , -2:-1]
            rho142 = np.max(np.linalg.eigvals(np.hstack((v72, v82))))
            if rho142 == 1:
                IRN2 = lamb12
                    
        while rho142 > 1:
            lamb12 += .0001
            sol52 = solve_ivp(ode32, [t.min(), t.max()], x01, t_eval=t)
            v52 = sol52.y[: , -2:-1]
            sol62 = solve_ivp(ode32, [t.min(), t.max()], x02, t_eval=t)
            v62 = sol62.y[: , -2:-1]
            rho132 = np.max(np.linalg.eigvals(np.hstack((v52, v62))))

            sol72 = solve_ivp(ode22, [t.min(), t.max()], x01, t_eval=t)
            v72 = sol72.y[: , -2:-1]
            sol82 = solve_ivp(ode22, [t.min(), t.max()], x02, t_eval=t)
            v82 = sol82.y[: , -2:-1]
            rho142 = np.max(np.linalg.eigvals(np.hstack((v72, v82))))
            if rho142 == 1:
                IRN2 = lamb12

    if IRN2 == 1:
        IRN2list.append(IRN2)
        sol1 = solve_ivp(ode1, [t.min(), t.max()], x01, t_eval=t)
        v1 = sol1.y[:, -2:-1]
        sol2 = solve_ivp(ode1, [t.min(), t.max()], x02, t_eval=t)
        v2 = sol2.y[:, -2:-1]
        rho11 = np.max(np.linalg.eigvals(np.hstack((v1, v2))))
        BRN1 = lamb

        while rho11 > 1:
            lamb += .0001
            sol1 = solve_ivp(ode1, [t.min(), t.max()], x01, t_eval=t)
            v1 = sol1.y[:, -2:-1]
            sol2 = solve_ivp(ode1, [t.min(), t.max()], x02, t_eval=t)
            v2 = sol2.y[:, -2:-1]
            rho11 = np.max(np.linalg.eigvals(np.hstack((v1, v2))))
            BRN1 = lamb
    
        while rho11 < 1:
            lamb -= .0001
            sol1 = solve_ivp(ode1, [t.min(), t.max()], x01, t_eval=t)
            v1 = sol1.y[:, -2:-1]
            sol2 = solve_ivp(ode1, [t.min(), t.max()], x02, t_eval=t)
            v2 = sol2.y[:, -2:-1]
            rho11 = np.max(np.linalg.eigvals(np.hstack((v1, v2))))
            BRN1 = lamb
    
        if rho11 == 1:
            BRN1 = lamb
                
        sol12 = solve_ivp(ode12, [t.min(), t.max()], x01, t_eval=t)
        v12 = sol12.y[:, -2:-1]
        sol22 = solve_ivp(ode12, [t.min(), t.max()], x02, t_eval=t)
        v22 = sol22.y[:, -2:-1]
        rho112 = np.max(np.linalg.eigvals(np.hstack((v12, v22))))
        BRN2 = lamb2

        while rho112 > 1:
            lamb2 += .0001
            sol12 = solve_ivp(ode12, [t.min(), t.max()], x01, t_eval=t)
            v12 = sol12.y[:, -2:-1]
            sol22 = solve_ivp(ode12, [t.min(), t.max()], x02, t_eval=t)
            v22 = sol22.y[:, -2:-1]
            rho112 = np.max(np.linalg.eigvals(np.hstack((v12, v22))))
            BRN2 = lamb2
    
        while rho112 < 1:
            lamb2 -= .0001
            sol12 = solve_ivp(ode12, [t.min(), t.max()], x01, t_eval=t)
            v12 = sol12.y[:, -2:-1]
            sol22 = solve_ivp(ode12, [t.min(), t.max()], x02, t_eval=t)
            v22 = sol22.y[:, -2:-1]
            rho112 = np.max(np.linalg.eigvals(np.hstack((v12, v22))))
            BRN2 = lamb2
    
        if rho112 == 1:
            BRN2 = lamb2
                

    while IRN2 > 1:
        beta_h2 -= .0001
            
        sol52 = solve_ivp(ode32, [t.min(), t.max()], x01, t_eval=t)
        v52 = sol52.y[: , -2:-1]
        sol62 = solve_ivp(ode32, [t.min(), t.max()], x02, t_eval=t)
        v62 = sol62.y[: , -2:-1]
        rho132 = np.max(np.linalg.eigvals(np.hstack((v52, v62))))

        sol72 = solve_ivp(ode22, [t.min(), t.max()], x01, t_eval=t)
        v72 = sol72.y[: , -2:-1]
        sol82 = solve_ivp(ode22, [t.min(), t.max()], x02, t_eval=t)
        v82 = sol82.y[: , -2:-1]
        rho142 = np.max(np.linalg.eigvals(np.hstack((v72, v82))))
        IRN2 = lamb12
            
        while rho142 < 1:
            lamb12 -= .0001
            sol52 = solve_ivp(ode32, [t.min(), t.max()], x01, t_eval=t)
            v52 = sol52.y[: , -2:-1]
            sol62 = solve_ivp(ode32, [t.min(), t.max()], x02, t_eval=t)
            v62 = sol62.y[: , -2:-1]
            rho132 = np.max(np.linalg.eigvals(np.hstack((v52, v62))))

            sol72 = solve_ivp(ode22, [t.min(), t.max()], x01, t_eval=t)
            v72 = sol72.y[: , -2:-1]
            sol82 = solve_ivp(ode22, [t.min(), t.max()], x02, t_eval=t)
            v82 = sol82.y[: , -2:-1]
            rho142 = np.max(np.linalg.eigvals(np.hstack((v72, v82))))
            if rho142 == 1:
                IRN2 = lamb12
                    
        while rho142 > 1:
            lamb12 += .0001
            sol52 = solve_ivp(ode32, [t.min(), t.max()], x01, t_eval=t)
            v52 = sol52.y[: , -2:-1]
            sol62 = solve_ivp(ode32, [t.min(), t.max()], x02, t_eval=t)
            v62 = sol62.y[: , -2:-1]
            rho132 = np.max(np.linalg.eigvals(np.hstack((v52, v62))))

            sol72 = solve_ivp(ode22, [t.min(), t.max()], x01, t_eval=t)
            v72 = sol72.y[: , -2:-1]
            sol82 = solve_ivp(ode22, [t.min(), t.max()], x02, t_eval=t)
            v82 = sol82.y[: , -2:-1]
            rho142 = np.max(np.linalg.eigvals(np.hstack((v72, v82))))
            if rho142 == 1:
                IRN2 = lamb12

    if IRN2 == 1:
        IRN2list.append(IRN2)
        sol1 = solve_ivp(ode1, [t.min(), t.max()], x01, t_eval=t)
        v1 = sol1.y[:, -2:-1]
        sol2 = solve_ivp(ode1, [t.min(), t.max()], x02, t_eval=t)
        v2 = sol2.y[:, -2:-1]
        rho11 = np.max(np.linalg.eigvals(np.hstack((v1, v2))))
        BRN1 = lamb

        while rho11 > 1:
            lamb += .0001
            sol1 = solve_ivp(ode1, [t.min(), t.max()], x01, t_eval=t)
            v1 = sol1.y[:, -2:-1]
            sol2 = solve_ivp(ode1, [t.min(), t.max()], x02, t_eval=t)
            v2 = sol2.y[:, -2:-1]
            rho11 = np.max(np.linalg.eigvals(np.hstack((v1, v2))))
            BRN1 = lamb
    
        while rho11 < 1:
            lamb -= .0001
            sol1 = solve_ivp(ode1, [t.min(), t.max()], x01, t_eval=t)
            v1 = sol1.y[:, -2:-1]
            sol2 = solve_ivp(ode1, [t.min(), t.max()], x02, t_eval=t)
            v2 = sol2.y[:, -2:-1]
            rho11 = np.max(np.linalg.eigvals(np.hstack((v1, v2))))
            BRN1 = lamb
    
        if rho11 == 1:
            BRN1 = lamb
                
        sol12 = solve_ivp(ode12, [t.min(), t.max()], x01, t_eval=t)
        v12 = sol12.y[:, -2:-1]
        sol22 = solve_ivp(ode12, [t.min(), t.max()], x02, t_eval=t)
        v22 = sol22.y[:, -2:-1]
        rho112 = np.max(np.linalg.eigvals(np.hstack((v12, v22))))
        BRN2 = lamb2

        while rho112 > 1:
            lamb2 += .0001
            sol12 = solve_ivp(ode12, [t.min(), t.max()], x01, t_eval=t)
            v12 = sol12.y[:, -2:-1]
            sol22 = solve_ivp(ode12, [t.min(), t.max()], x02, t_eval=t)
            v22 = sol22.y[:, -2:-1]
            rho112 = np.max(np.linalg.eigvals(np.hstack((v12, v22))))
            BRN2 = lamb2
    
        while rho112 < 1:
            lamb2 -= .0001
            sol12 = solve_ivp(ode12, [t.min(), t.max()], x01, t_eval=t)
            v12 = sol12.y[:, -2:-1]
            sol22 = solve_ivp(ode12, [t.min(), t.max()], x02, t_eval=t)
            v22 = sol22.y[:, -2:-1]
            rho112 = np.max(np.linalg.eigvals(np.hstack((v12, v22))))
            BRN2 = lamb2
    
        if rho112 == 1:
            BRN2 = lamb2
            
    IRN2list.append(IRN2)
    resultIRN2BRN1.append(BRN1)
    resultIRN2BRN2.append(BRN2)

plt.plot(resultIRN2BRN2, resultIRN2BRN1, label="IRN2")
plt.xlabel("BRN(beta_h1)")
plt.ylabel("BRN(beta_h2)")
plt.show()
