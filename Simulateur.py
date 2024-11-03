import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from voronoi import isAvoidanceNeeded, isInCircle, thinPolygon, buildVoronoiDiagram, bestDirFunction
import time

#%% ------------------------------ Affichage ------------------------------ %%#

def display(facets, obstacles, centers, nextPoint, boat, waypoint, pointToCheck, parameters):
    ax.clear()

    for f in facets:
        ax.plot([p[0] for p in f], [p[1] for p in f], linewidth = 1, color = 'lightblue')           # Affichage des segment du diagramme de Voronoi

    thinnedPolygon = thinPolygon(facets[0], 0.5)
    ax.plot([p[0] for p in thinnedPolygon] + [thinnedPolygon[0][0]], [p[1] for p in thinnedPolygon] + [thinnedPolygon[0][1]], linewidth = 1, color = 'red')           # Affichage des segment du diagramme de Voronoi aminci
    
    p_score = []
    for p in pointToCheck:
        p_score.append(bestDirFunction(p, centers, boat, waypoint, parameters))

    max_score = max(p_score)
    plt.scatter([p[0] for p in pointToCheck], [p[1] for p in pointToCheck], c=[(s/max_score, 0, 1-s/max_score) for s in p_score])

    ax.scatter(boat.x, boat.y, 10, color="green")    # Affichage du bateau
    drawCircle(boat.position, 1, "red")
    drawCircle(boat.position, 4, "orange")
    # for p in centers[1::]:
    ax.scatter([p[0] for p in centers[1::]], [p[1] for p in centers[1::]], 20, color="red") # Affichage des obstacles

    for obst in obstacles:
        if not isInCircle(obst.position, boat.position, 4):
            ax.scatter(obst.x, obst.y, 20, color="gray") # Affichage des obstacles


    ax.plot([boat.x, waypoint[0]], [boat.y, waypoint[1]], "--",linewidth = 1, color = "gray") # Ligne de cap jusqu'au prochain waypoint
    ax.axis([boat.x - 10, boat.x + 10, boat.y - 10, boat.y + 10])    # Restreint l'affichage à la zone où sont les obstacles
    
    ax.scatter(waypoint[0], waypoint[1], 20, color = "black")
    ax.scatter(nextPoint[0], nextPoint[1], 25, "lime")

def drawCircle(center, radius, clr):
    x = center[0] + radius*np.cos(np.arange(0,2*np.pi + 0.1, 0.1))
    y = center[1] + radius*np.sin(np.arange(0,2*np.pi + 0.1, 0.1))

    ax.plot(x,y, '--', color = clr, linewidth=1, alpha=0.6)


#%% ------------------------------ Class ------------------------------ %%# 

class Boat:
    def __init__(self, position : np.array, cap : float, speed : float) -> None:
        self.position = np.array(position, dtype=float)
        self.x = self.position[0]
        self.y = self.position[1]
        self.cap = cap
        self.speed = speed

    # Affichage only
    def move(self, dt):
        self.position += dt * self.speed * np.array([np.cos(self.cap), np.sin(self.cap)])
        self.x = self.position[0]
        self.y = self.position[1]
    
    def newCap(self, target):
        """
        Definition du nouveau cap avec une angle absolu
        """
        if target[1] < self.position[1]:
            if target[0] < self.position[0]:
                self.cap =np.pi + np.arctan((target[1] - self.position[1])/(target[0] - self.position[0]))
                return
            elif target[0] > self.position[0]:
                self.cap = 2*np.pi + np.arctan((target[1] - self.position[1])/(target[0] - self.position[0]))
                return
        elif target[0] > self.position[0]:
            self.cap = np.arctan((target[1] - self.position[1])/(target[0] - self.position[0]))
            return
        else:
            self.cap = np.pi + np.arctan((target[1] - self.position[1])/(target[0] - self.position[0]))
            return
        
        if target[0] == self.position[0]:
            if target[1] < self.position[1]:
                self.cap = 3 * np.pi / 2
            else:
                self.cap = np.pi / 2

class Obstacle:
    """
    ## Description
        Classe Obstacles

    ### Constructeur
        position (np.array): Position de l'obstacle dans le plan cartésien
        speed (np.array): Vitesse de l'obstacle dans le plan cartésien
        size (float): Taille de l'obstacle
        priority (int): Priorite de l'obstacle (0: non prioritaire, 1: prioritaire, 2: non-manoeuvrant)
    """
    def __init__(self, position : np.array, speed : np.array, size : float, priority : int) -> None:
        """
        ## Description
            Constructeur de la classe Obstacle
            
        ### Args:
            position (np.array): Position de l'obstacle dans le plan cartésien
            speed (np.array): Vitesse de l'obstacle dans le plan cartésien
            size (float): Taille de l'obstacle
            priority (int): Priorite de l'obstacle (0: non prioritaire, 1: prioritaire, 2: non-manoeuvrant)
        """
        self.position = np.array(position, dtype=float)
        self.x = position[0]
        self.y = position[1]
        self.speed = np.array(speed)
        self.priority = priority
        self.size = size

    def move(self, dt):
        """
        ## Description
            Fonction de déplacement de l'objet (Fictive)
            
        ### Args:
            dt (float): Pas de temps pour le déplacement de l'obstacle
        """
        self.position += dt * self.speed
        self.x = self.position[0]
        self.y = self.position[1]


#%% ------------------------------ Fonctions ------------------------------ %%#
            
def moveObject(obstacles, boat : Boat, dt):
    
    boat.move(dt)
    for obst in obstacles:
        obst.move(dt)

#%% ------------------------------ Initialisation ------------------------------ %%#

boat = Boat([17.0,17.0], 0, 1)
obstacles = [Obstacle([np.random.rand()*14+2, np.random.rand()*14+2], 0, 0.2, 0) for _ in range(30)]
waypoint = (1, 1)
    
fig, ax = plt.subplots()

f = open('data.txt', 'w')
f.close()

#%% ------------------------------ Program ------------------------------ %%#

T = []
def update(_):
    if isInCircle(boat.position, waypoint, 0.1):
        anim.pause()
        plt.close(fig)

    start = time.perf_counter()
    parameters = {"wind" : (0,1), "current" : (3,0.2)}
    facets, centers = buildVoronoiDiagram(boat, obstacles)
    endVoronoi = time.perf_counter()
    (nextPoint, pointToCheck) = isAvoidanceNeeded(facets, centers, boat, waypoint, parameters)
    end = time.perf_counter()

    print(f"Build Voronoi diagram execution time : {endVoronoi - start:.3e} s")
    print(f"Is avoidance needed execution time : {end - endVoronoi:.3e} s\n")
    T.append(end - start)
    
    boat.newCap(nextPoint)
    moveObject(obstacles, boat, 0.05)

    # Affichage seulement
    display(facets, obstacles, centers, nextPoint, boat, waypoint, pointToCheck, parameters)
    # 

anim = animation.FuncAnimation(fig, update, interval= 1000/20)
manager = plt.get_current_fig_manager()
manager.full_screen_toggle()
plt.show()

print(f"Temps moyen : {sum(T)/len(T):.3e} s")


#%% ------------------------------ Remarques ------------------------------ %%#
"""
    Les polygones d'un diagramme de Voronoi sont convexe
    # http://www.mathom.fr/mathom/sauvageot/Modelisation/Graphes/Voronoi_telephone.pdf



"""