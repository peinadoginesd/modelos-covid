# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 02:22:24 2020

@author: daniel
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Total population, N.
N = 6.6e+6
# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = 4./N, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = 1 - I0 - R0
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
RCero, gamma = 4, 1./15
beta = RCero * gamma

# A grid of time points (in days)
t = np.linspace(0, 150, 150)

# The SIR model differential equations.
def deriv(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Initial conditions vector
y0 = S0, I0, R0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(beta, gamma))
S, I, R = ret.T

# Plot the data on three separate curves for S(t), I(t) and R(t)
plt.style.use(plt.style.available[14])
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, axisbelow=True) # axis_bgcolor='#dddddd'
ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.set_xlabel('tiempo /days')
ax.set_ylabel('Número')
ax.set_ylim(0,1.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()