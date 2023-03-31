#%%
import numpy as np
import matplotlib.pyplot as plt
import time
##load data
RVtype = np.dtype({
    'names':['date', 'rv', 'e'],
    'formats':['i','f', 'f']})
D = np.loadtxt('data.txt', dtype=RVtype)
N = len(D)    #samples
D["date"] = D["date"]-2450681.527700
plt.scatter(D["date"],D["rv"])
#%%
##function
def f(t,params):
    params = params.reshape(1,-1)
    phi = 2*t*np.pi/params[0][3]
    return params[0][2] + params[0][0]*np.sin(phi) + params[0][1]*np.cos(phi)

##matrix operator
def dot(A,B):
    #A=A.reshape(1,-1)
    rA = A.shape[0]
    cA = B.shape[0]
    cB = B.shape[1]
    sum = np.zeros((rA,cB))
    for j in range(cA):
        for i in range(rA):
            for k in range(cB):
                sum[i][k] += A[i][j] * B[j][k]
    return sum
##solve linear equation for theta+
def solve(A,B):
    A_inv = np.linalg.inv(A)
    x = dot(A_inv,B)
    return x
##gradient
def grad(theta,S):
    theta = theta.reshape(1,-1)
    grad_f = np.zeros((N,theta.shape[1]))
    for i in range(N):
        t = S["date"][i]
        grad_f[i][0] = np.sin((2*np.pi*t)/theta[0][3])
        grad_f[i][1] = np.cos((2*np.pi*t)/theta[0][3])
        grad_f[i][2] = 1
        grad_f[i][3] = (-theta[0][0]*np.cos((2*np.pi*t)/theta[0][3]) + theta[0][1]*np.sin((2*np.pi*t)/theta[0][3])) * (2*np.pi*t) / np.square(theta[0][3])
    return grad_f
##Hessian
def hess(theta,S):
    theta = theta.reshape(1,-1)
    G = grad(theta,S)
    H = np.zeros((theta.shape[1],theta.shape[1]))
    for i in range(N):
        for j in range(theta.shape[1]):
            for k in range(theta.shape[1]):
                H[j][k] = G[i][j]*G[i][k] / np.square(S["e"][i])
    return H
##first order derivative of chi_square
def chi_derivative(theta,S):
    diff = (f(S["date"],theta) - S["rv"]) / np.square(S["e"])
    diff = diff.reshape((1,N))
    theta = theta.reshape(1,-1)
    grad_f = grad(theta,S)
    return dot(diff,grad_f)
##chi_square
def chi_square(theta,S):
    chi_sqr = 0
    for i in range(N):
        chi_sqr += np.square( ( f(S["date"][i],theta) - S["rv"][i] ) / S["e"][i] )
    return chi_sqr
##Levenberg-Marquardt Method
def Mrq(theta,S,lim):
    start=time.time()
    check=start
    chi_sqr0 = chi_square(theta,S)
    l = .00001
    while (chi_sqr0 >= lim)&(check-start<10):  #pause criteria
        B = -chi_derivative(theta,S)
        A = hess(theta,S)
        #A = dot(B.T,B)
        for i in range(A.shape[0]):
            A[i][i] = A[i][i] * (1+l)
        theta_1 = theta + solve( A , B.T ).T
        chi_sqr1 = chi_square(theta_1,S)
        if (chi_sqr1 >= chi_sqr0):
            l = l * 10
        else:
            l = l / 10
            theta = theta_1 
            chi_sqr0 = chi_sqr1 
        #print(theta,chi_sqr0)
        check=time.time()
    return theta


def Search():
    limit=5e2

    Bound_A=200
    Bound_B=200
    Bound_m=40
    Bound_P=1400
    A=-200
    while A<Bound_A:
        B=-200
        while B<Bound_B:
            m=-40
            while m<Bound_m:
                P=2
                while P<Bound_P:
                    p0 = np.array([A,B,m,P])
                    pf,chi_sqr = Mrq(p0,D,limit)
                    print(pf,chi_sqr)
                    P+=50
                m+=10
            B+=30
        A+=30
    return 0
#Search()
##fit
plt.xlim(300,850)
plt.scatter( D["date"] , D["rv"] ,c="r",label='observed')
x = np.linspace(0,D["date"].max(),10000)
par = np.array([ 43.28144963,61.0055289,  -10.28913117  , 3.52427464])
#par = Mrq( par , D , 200)
print(par,chi_square(par,D))
plt.plot(x,f(x,par),linewidth=0.9,label='fit')
plt.legend()
# %%
