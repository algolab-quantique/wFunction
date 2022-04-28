import quantit as qtt
import torch
import interpolate as terp
import jax.numpy as jnp
import numpy as np
import seaborn as sns
sns.set_theme()
import matplotlib.pyplot as plt

if __name__=='__main__':
    def f(x):
        return jnp.exp(-x**2/2)
    degree = 4
    C_full = np.polynomial.chebyshev.Chebyshev.interpolate(f,degree,domain=(-1,1))
    C_left = np.polynomial.chebyshev.Chebyshev.interpolate(f,degree,domain=(-1,0))
    C_right = np.polynomial.chebyshev.Chebyshev.interpolate(f,degree,domain=(0,1))

    w = np.linspace(-1,1,2*100000 )
    w_right = np.linspace(0,1,100000)
    w_left = np.linspace(-1,0,100000)
    dw = w_right[1]-w_right[0]
    rms = np.sqrt(np.sum((C_full(w_left)-C_left(w_left))**2*dw + (C_full(w_right) - C_right(w_right))**2*dw))
    print("left edge: ",C_full(-1)-C_left(-1))
    print("right edge:", C_full(1)-C_right(1))
    print("rms: ", rms)
    plt.plot(w,f(w),label='actual')
    plt.plot(w,C_full(w), label="full_C")
    plt.plot(w_left,C_left(w_left), label="left")
    plt.plot(w_right,C_right(w_right), label="right")
    plt.legend()
    plt.show()

