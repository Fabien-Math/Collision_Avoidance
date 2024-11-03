import matplotlib.pyplot as plt
import numpy as np

""""Ce script permet de lire les données écrite dans un fichier par le script Voronoi"""

Data = {'x': [], 'y': [], 'p1': [], 'p2': [], 'p3': [], 'p4': [], 'p5': [], 'p6': [], 'cap': [], 'angle': []}
name = ['x', 'y', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'cap', 'angle']

with open("data.txt", 'r') as f:
    for l in f:
        d = l.split('\t')
        if d != ['\n'] and len(d)>2:
            for i in range(8):
                Data[name[i]].append(float(d[i]))
        else:
            break

X = Data['p1']
Y = Data['p2']

# plt.plot(X, Y)
# plt.show()

print(Data['x'], Data['y'], Data['p6'])
plt.scatter(Data['x'], Data['y'], 50, c=[[x6/360, 0, 1-x6/360] for x6 in Data['p6']])
plt.scatter(8, 8, 50, c=[0,0,0])
# plt.scatter(8, 8*sin(), 50, c=[0,0,0])
plt.show()


X = np.arange(0, 1.01, 0.01)
Y = np.arange(0, 1.01, 0.01)

X, Y = np.meshgrid(X, Y)
Z = 2*np.cos(X+Y)


plt.contourf(X, Y, Z, 1000)
plt.axis('equal')
plt.axis([-0.1, 1.1, -0.1, 1.1])
plt.show()