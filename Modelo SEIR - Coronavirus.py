# -*- coding: utf-8 -*-
"""
MODELO SEIR COVID-19
Basado en la transmisión S -> E -> I -> R, donde:
    
    S: población susceptible.
    E: población en periodo de incubación ó infectados asintomáticos.
    I: población infectada con síntomas (fuente del colapso sanitario).
    R: población recuperada con inmunidad PERMANENTE.

Obsérvese que se han considerado las siguientes suposiciones:
    
    - La inmunidad de los recuperados es permanente.
    - El número reporductivo básico y la tasa de contagio es constante.
    - Los datos medios de la enfermedad para una persona media son:
        
    Tiempo de Incubación -> 6 días
    Tiempo de Recuperación -> 14 días
    Tasa de Mortalidad -> 5%
    
"""
print (__doc__)

import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import datetime as dt
from math import log

# Fechas.
dateBegin, dateLast = dt.date(2020, 3, 22), 150 #dt.date(2020, 2, 8), 150
meses = ['ENE', 'FEB', 'MAR', 'ABR', 'MAY', 'JUN', 'JUL', 'AGO', 'SEP', 'OCT', 
         'NOV', 'DIC']

# Condiciones iniciales.
N = 23e+6 #45e+6 # población total inicial.
I0, E0, R0, M0 = 30000./N, 20000./N, 3000./N, 0.
S0 = 1. - (I0 + E0 + R0 + M0)

# Parámetros: gamma recuperación, mu mortalidad, epsilon incubación (por día).
RZero = 4 # min 1.4 max 4.08.
gamma, mu, epsilon = 1./14, 0.05/14, 1./6
beta = RZero * gamma

# =============================================================================

t = np.arange(0, dateLast + 1) # vector de tiempos (en días).
# Modelo SEIR con ecuaciones diferenciales.
def deriv(y, t, beta, gamma, mu, epsilon):
    S, E, I, R, M = y
    dSdt = -beta * S * (I + E)
    dEdt = beta * S * (I + E) - epsilon * E
    dIdt = epsilon * E - (gamma + mu) * I
    dRdt = gamma * I
    dMdt = mu * I
    return dSdt, dEdt, dIdt, dRdt, dMdt

y0 = S0, E0, I0, R0, M0 # vector de condiciones iniciales.
# Integral de las ecuaciones sobre el vector de tiempos t.
ret = odeint(deriv, y0, t, args=(beta, gamma, mu, epsilon))
S, E, I, R, M = ret.T

plt.style.use('bmh')
# Plot de las curvas S(t), E(t), I(t) y R(t).
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, axisbelow=True)

ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptibles')
ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infectados sintomáicos')
ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recuperados con inmunidad')
ax.plot(t, E, 'grey', alpha=0.5, lw=1, label='Portadores asintomáticos o incubando')

ax.set_ylim(0,1.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)

step = int(dateLast/(len(ax.get_xticklabels())-3))
dates = [dateBegin + dt.timedelta(days=int(i)) for i in range(-step, dateLast+1)]
labelDates = ['%s-%s' %(date.day, meses[date.month - 1]) for k, date in 
              enumerate(dates) if (k % step == 0)]
ax.set_xticklabels(labelDates)

ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
fig.suptitle('Modelo SEIR: COVID-19', fontsize=14, fontweight='bold')
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)

"""   
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, axisbelow=True)
ax.plot(t, list(map(log, I)))
plt.show()
"""

# DataFrame con las predicciones:
data = pd.DataFrame({
        'Día': ['%s/%s' %(day.day, meses[day.month - 1]) for day in dates[step:]],
        'Infectados': np.around(I*N).astype(int), 
        'Muertos': np.around(M*N).astype(int),
        'Recuperados': np.around(R*N).astype(int),
        'Total Positivos': np.around(N*(I + R + M)).astype(int)
                    })
#data.to_excel('COVID-19_españa.xlsx', header=True, index=False, startcol=1)