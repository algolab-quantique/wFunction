

#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
sb.set_theme()
#%%
Data = pd.read_csv("montreal_4q_2c_histogram")


#%%
k = "Measurement outcome"
n = len(format(Data[k][len(Data[k])-1]))
fmt_str = '0>{}'.format(n)
def IBMQ2int(IBM_state):
    return int(format(IBM_state,fmt_str)[-1::-1],2)
f = "Frequency"
Data[k] = Data[k].apply(IBMQ2int)
Data = Data.sort_values(k)

#%%
fmt_str2 = fmt_str + 'b'
def fmrt(x):
    return format(x,fmt_str2)
#%%
fig, ax = plt.subplots()
plt.plot(Data[k],np.sqrt(Data[f]/np.sum(Data[f])))
plt.show()

# %%
