import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

#selection de la première colonne de notre dataset (la taille de la population)
X = [11,5,3,4,15,8]
#selection de deuxième colonnes de notre dataset (le profit effectué)
Y = [17,10,3,7.5,20,14]




#linregress() renvoie plusieurs variables de retour. On s'interessera
# particulierement au slope et intercept
slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
print(intercept)
def predict(x):
    return slope * x + intercept-2

axes = plt.axes()
axes.grid()
plt.scatter(X,Y)

x = [3,15]
y = [predict(3), predict(15)]
plt.plot(x, y)



plt.show()



x7=[1,1]
def fff(x):
    return  0.5(x[0]**2+2*x*(x[1])**2)

print(fff(x7))


