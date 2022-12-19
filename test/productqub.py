#%%
import sympy as sp



I,W = sp.symbols('I W')
t0,t1,t2,t3,t4 = sp.symbols(R'\theta_0 \theta_1 \theta_2 \theta_3 \theta_4' )
# %%

s = sp.Matrix([[1],[0]])
def zry(theta):
    return sp.Matrix([[sp.cos(theta),sp.sin(theta)],[sp.sin(theta),-sp.cos(theta)]])

cW = sp.Matrix([[1,0],[0,W]])
H = sp.Matrix([[1,1],[1,-1]])
#%%
s0 = cW@zry(t0)@s
s0
# %%
s1 = cW@zry(t1)@s0
s1
# %%
H@s1
# %%
