#%%
import numpy as np
import matplotlib.pyplot as plt
##load data
RVtype = np.dtype({
    'names':['date', 'rv', 'e'],
    'formats':['i','f', 'f']})
D = np.loadtxt('data.txt', dtype=RVtype)
N = len(D)    #samples
D["date"] = D["date"]

##P_N
def LombPeriodogram(D,omega):
    su,sl = 0,0
    for i in range(N):
        su += np.sin( 2*omega*D["date"][i] )
        sl += np.cos( 2*omega*D["date"][i] )
    tangent_2omegatau = su / sl
    tau = np.arctan(tangent_2omegatau)
    scu,scl,ssu,ssl = 0,0,0,0
    h_bar = np.average(D["rv"])
    sigma_sqr = np.std(D["rv"])
    P_N = 0
    for i in range(N):
        scu += ( (D["rv"][i]-h_bar)*np.cos(omega*(D["date"][i]-tau)) )
        scl += np.square( np.cos(omega*(D["date"][i]-tau)) )
        ssu += ( (D["rv"][i]-h_bar)*np.sin(omega*(D["date"][i]-tau)) )
        ssl += np.square( np.sin(omega*(D["date"][i]-tau)) )
    print(ssl)
    P_N = ( np.square(scu)/ scl + np.square(ssu)/ ssl )/(2*sigma_sqr)
    return P_N
##calculate
f = np.linspace(0.0001,4,100000)
x = f*2*np.pi
y = LombPeriodogram(D,x)
print(y)
plt.plot(x,y)
# %%
