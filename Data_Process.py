#%%
import numpy as np
##load data
RVtype = np.dtype({
    'names':['date', 'rv', 'e'],
    'formats':['i','f', 'f']})
D = np.loadtxt('data.txt', dtype=RVtype)
N = len(D)    #samples
D_0 = D["date"][0]
D["date"] = D["date"] - D_0
##write data
np.savetxt('data_out.txt',D["rv"])

# %%
