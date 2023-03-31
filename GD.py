#%%
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
##load data
RVtype = np.dtype({
    'names':['date', 'rv', 'e'],
    'formats':['i','f', 'f']})
D = np.loadtxt('data.txt', dtype=RVtype)
N = len(D)    #samples
D["date"] = D["date"] - 2450681.527700
plt.scatter(D["date"],D["rv"])

alpha=.4
delta_theta = np.array([[.0001,0,0,0],[0,.0001,0,0],[0,0,0.0001,0],[0,0,0,.001]])
##RV function
def f(t,params):
    params = params.reshape(1,-1)
    phi = 2*t*np.pi/params[0][3]
    return params[0][2] + params[0][0]*np.sin(phi) + params[0][1]*np.cos(phi)
##matrix operator
def dot(A,B):
    A=A.reshape(1,-1)
    rA = A.shape[0]
    cA = A.shape[1]
    cB = B.shape[1]
    sum = np.zeros((rA,cB))
    for j in range(cA):
        for i in range(rA):
            for k in range(cB):
                sum[i][k] += A[i][j] * B[j][k]
    return sum
##gradient descent
def grad(theta,S):
    diff = (f(S["date"],theta) - S["rv"])/S["e"]
    diff = diff.reshape((1,N))
    grad_f = np.zeros((N,theta.shape[0]))
    for i in range(N):
        for j in range(theta.shape[0]):
            grad_f[i][j] = (f(S["date"][i],theta+delta_theta[j])-f(S["date"][i],theta)) / delta_theta[j][j]
    return dot(diff,grad_f)
def gradient_descent(theta,S):
    local_gradient = grad(theta,S)
    while not np.all(np.abs(local_gradient) <= 1e-5):
        theta = theta - alpha * local_gradient
        local_gradient = grad(theta,S)
    return theta
def chi_square(theta,S):
    chi_sqr = 0
    for i in range(N):
        chi_sqr += np.square( ( f(S["date"][i],theta) - S["rv"][i] ) / S["e"][i] )
    return chi_sqr
##fit
p0 = np.array([60,60,-5,600])
pf = gradient_descent(p0,D)
print(pf)
plt.scatter(D["date"],f(D["date"],pf))
x=np.linspace(0,2000,10000)
plt.plot(x,f(x,pf))
plt.show()
##deviation
plt.plot(x,x*0)
plt.errorbar(D["date"],f(D["date"],pf)-D["rv"],yerr=D["e"])
print(alpha,chi_square(pf,D))
#print(delta_theta)


# %%
