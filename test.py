import numpy as np
import matplotlib.pyplot as plt
from util import convex_hull


# Calcul et renvoie le point d'intersection de deux droites avec pour équation de droite : a1 * x + b1 * y + c1 = 0 et a2 * x+ b2 * y + c2 = 0
def findIntersection(a1 : float, b1 : float, c1 : float, a2 : float, b2 : float, c2 : float) -> list[float, float]:
    if a1 * b2 == a2 * b1:
        print("Warning - findIntersection\n Les deux droites sont colinéaires")
        return None
        # raise "Fatal Error - findIntersection\n Les deux droites sont colinéaires"

    # Resultat trouvé en résolvant le système (a1 * x + b1 * y + c1 = 0; a2 * x + b2 * y + c2 = 0)
    x = (b1 * c2 - c1 * b2)/(a1 * b2 - a2 * b1) 
    y = (a1 * c2 - a2 * c1)/(a2 * b1 - a1 * b2)
    return (x,y)

# Calcul et renvoie la distance entre deux points
def euclidean_distance(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

# Calcul et renvoie l'équation 'ax + by + c = 0' d'une ligne à partir des deux points 'p1' et 'p2'
def findLineEq(p1 : list[float], p2 : list[float]) -> list[float, float, float]:
    if p1[0] == p2[0]:  # Si la droite est verticale, x = c
        return 1, 0, -p1[0]
    # y = -a/b * x - (p1[1] - (a / b) * p1[0]) => (On multiplie tout par b) => by = -ax - c avec c = p1[1]*b + a*p1[0]
    a = p1[1] - p2[1]
    b = p2[0] - p1[0]
    c = - a * p1[0] - b * p1[1]
    return a, b, c

eq =  [(-0.25, 1, -6.5), (0, 1, -8.5), (3, 1, -30)]
# eq =  [(-0.25, 1, -6.5), (0, 1, -8.5), (1, 0 , -8.5)]

xc = 8
yc = 8
p_c = (xc, yc)
r = 2
X = r*np.cos(np.linspace(0, 2*np.pi, 100))+xc
Y = r*np.sin(np.linspace(0, 2*np.pi, 100))+yc

points = []
for x, y in zip(X, Y):
    ac, bc, cc = findLineEq(p_c, (x,y))
    temp = [(x,y)]
    for a,b,c in eq:
        point = findIntersection(a,b,c,ac,bc,cc)
        if a == 0:
            print(point, x, y)
        if point != None and euclidean_distance(point, p_c) < 2 and np.sign((x-xc)*(point[0]-xc)) >= 0 and np.sign((y-yc)*(point[1]-yc)) >= 0:
            temp.append(point)
    
    
    min_p = temp[0]
    for p in temp:
        if euclidean_distance(p, p_c) < euclidean_distance(min_p, p_c):
            min_p = p
    
    points.append(min_p)


# for i,x in enumerate(X[:50]):
#     val = []
#     for a,b,c in eq:
#         coord = findIntersection(a,b,c,1,0,-x)
#         if coord[1] >= 0:
#             val.append(coord[1])

#     if val != []:
#         min_y = val[0]
#         for j,v in enumerate(val):
#             if euclidean_distance((x,v), (xc,yc)) < euclidean_distance((x,min_y), (xc, yc)):
#                 min_y = v
#         points.append(min(min_y, Y[i]))
#     else:
#         points.append(Y[i])

# for i,x in enumerate(X[50::]):
#     val = []
#     for a,b,c in eq:
#         coord = findIntersection(a,b,c,1,0,-x)
#         if coord[1] < 0:
#             val.append(coord[1])
#     if val != []:
#         max_y = val[0]
#         for j,v in enumerate(val):
#             if euclidean_distance((x,v), (xc,yc)) < euclidean_distance((x, max_y), (xc, yc)):
#                 max_y = v
#         points.append(max(max_y, Y[i]))   
#     else:
#         points.append(Y[i+50])   

# plt.figure(0)
# points = []
# plt.scatter(X, Y)

# for i in range(len(eq)):
#     a1, b1, c1 = eq[i]
#     a2, b2, c2 = eq[(i+1)%len(eq)]

#     points.append(findIntersection(a1, b1, c1, a2, b2, c2))




# for i, (a,b,c) in enumerate(eq):
#     c_prime = -(c+a*xc+b*yc)
#     if r**2*(a**2 + b**2) - c_prime**2 > 0:
#         xp = (a*c_prime + b*np.sqrt(r**2*(a**2+b**2) - c_prime**2))/(a**2 + b**2)
#         xm = (a*c_prime - b*np.sqrt(r**2*(a**2+b**2) - c_prime**2))/(a**2 + b**2)
#         yp = (b*c_prime - a*np.sqrt(r**2*(a**2+b**2) - c_prime**2))/(a**2 + b**2)
#         ym = (b*c_prime + a*np.sqrt(r**2*(a**2+b**2) - c_prime**2))/(a**2 + b**2)
#         plt.plot((xm+xc, xp+xc), (ym+yc, yp+yc), marker = 'o')

#     points.append((xm+xc, ym+yc))
#     points.append((xp+xc, yp+yc))

#     if b:
#         X_f = np.linspace(xc-2, xc+2, 50)
#         Y_f = [(-a*x-c)/b for x in X_f]
#     else:
#         if a:
#             Y_f = np.linspace(xc-2, xc+2, 50)
#             X_f = [-c/a] * 50
#         else:
#             exit("Error")
#     print(a,b,c)
#     print(X_f, Y_f)
#     plt.scatter(X_f,Y_f)


# plt.scatter(xc,yc)
# plt.axis("equal")
# plt.figure(1)
# hull = convex_hull(points)
plt.scatter([t[0] for t in points], [t[1] for t in points], marker='o')
# plt.plot([p[0] for p in hull], [p[1] for p in hull], marker='o')
plt.scatter(xc, yc)

plt.axis("equal")
plt.show()