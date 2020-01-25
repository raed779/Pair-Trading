import matplotlib.pyplot as plt
import numpy as np
"""fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
x = np.linspace(-5,5,100)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.plot(x, 2*x+1, '-r', label='y=2x+1')
plt.plot(x, 2*x-1,'-.g', label='y=2x-1')
plt.plot(x, 2*x+3,':b', label='y=2x+3')
plt.plot(x, 2*x-3,'--m', label='y=2x-3')
plt.legend(loc='upper left')
plt.show()"""

"""
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = fig.add_subplot()

X = np.linspace(0,1,1000)
Y = np.cos(X*20)

ax1.plot(X,Y)
ax1.set_xlabel(r"Original x-axis: $X$")

new_tick_locations = np.array([.2, .5, .9])

def tick_function(X):
    V = 1/(1+X)
    return ["%.3f" % z for z in V]


ax2.set_xlabel(r"Modified x-axis: $1/(1+X)$")
plt.show()
"""



plt.grid(True)
plt.plot([50,100,150,200], [2,3,7,10], "b", linewidth=0.8, marker="*")
plt.plot([50,100,150,200], [2,7,9,10], "g", linewidth=0.8, marker="+")
plt.axis([80, 180, 1, 10])
plt.annotate('Limite', xy=(150, 7), xytext=(150, 5.5),
arrowprops={'facecolor':'black', 'shrink':0.05} )
plt.annotate("Test",xy=(170, 7), xytext=(170, 5.5),
             arrowprops=dict(arrowstyle="<|-|>"))
plt.xlabel('Vitesse')
plt.ylabel('Temps')
plt.show()