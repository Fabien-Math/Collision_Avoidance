import numpy as np
from math import sin, cos
import cv2
from util import convex_hull
import time


# Calcul et renvoie le produit scalaire de deux vecteurs
def dot_product(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]

# Calcul et renvoie la distance entre deux points
def euclidean_distance(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

# Calcul et renvoie le point appartenant au polygone le plus proche d'un autre point 
def closest_point_on_polygon(polygon, point):
    closest_point = None
    closest_distance = float('inf')
    for i in range(len(polygon)):
        segment_start = polygon[i]
        segment_end = polygon[(i+1) % len(polygon)]
        current_point = point_on_segment(segment_start, segment_end, point)
        current_distance = euclidean_distance(current_point, point)
        if current_distance < closest_distance:
            closest_point = current_point
            closest_distance = current_distance
    return closest_point

# Test si le point trouvé est sur le segment et renvoie le point le plus proche sur le segment
def point_on_segment(segment_start, segment_end, point):
    segment_vec = [segment_end[i]-segment_start[i] for i in range(2)]
    point_vec = [point[i]-segment_start[i] for i in range(2)]
    segment_length = euclidean_distance(segment_start, segment_end)
    projection = dot_product(segment_vec, point_vec) / segment_length
    if projection < 0:
        return segment_start
    elif projection > segment_length:
        return segment_end
    else:
        return [segment_start[i] + projection * segment_vec[i]/segment_length for i in range(2)]

# Calcul et renvoie l'équation 'ax + by + c = 0' d'une ligne à partir des deux points 'p1' et 'p2'
def findLineEq(p1 : list[float], p2 : list[float]) -> list[float, float, float]:
    if p1[0] == p2[0]:  # Si la droite est verticale, x = c
        return 1, 0, -p1[0]
    # y = -a/b * x - (p1[1] - (a / b) * p1[0]) => (On multiplie tout par b) => by = -ax - c avec c = p1[1]*b + a*p1[0]
    a = p1[1] - p2[1]
    b = p2[0] - p1[0]
    c = - a * p1[0] - b * p1[1]
    return a, b, c

# Calcul et renvoie le point d'intersection de deux droites avec pour équation de droite : a1 * x + b1 * y + c1 = 0 et a2 * x+ b2 * y + c2 = 0
def findIntersection(a1 : float, b1 : float, c1 : float, a2 : float, b2 : float, c2 : float) -> list[float, float]:
    if a1 * b2 == a2 * b1:
        raise "Fatal Error - findIntersection\n Les deux droites sont colinéaires"
    # Resultat trouvé en résolvant le système (a1 * x + b1 * y + c1 = 0; a2 * x + b2 * y + c2 = 0)
    x = (b1 * c2 - c1 * b2)/(a1 * b2 - a2 * b1) 
    y = (a1 * c2 - a2 * c1)/(a2 * b1 - a1 * b2)
    return (x,y)

# Test et renvoie si le point 'p' est dans le cercle de centre 'center' et de rayon 'radius' (Simule le champs de vision du bateau)
def isInCircle(p : list[float, float], center : list[float, float], radius : float) -> bool:
    if euclidean_distance(p, center) <= radius:
        return True
    return False

# Calcul et renvoie la plus courte distance entre un point et une droite
def orthogonalProjection(p  : list[float], a : float, b : float, c : float) -> float:
    # "https://fr.wikipedia.org/wiki/Distance_d'un_point_%C3%A0_une_droite"
    return abs(a*p[0] + b*p[1] + c)/np.sqrt(a**2 + b**2)

# Calcul et revoie les coefficient de l'équation de la droite ax + by + c = 0 translatée de 'distance' parallele a une autre droite
def calculate_parallel_line(a, b, c, distance) -> list[float, float, float]:
    return (a, b, c + distance * (a ** 2 + b ** 2) ** 0.5)

# Renvoie si le point 'point' est dans de polygone ou à l'exterieur
def point_in_polygon(point, polygon) -> bool:
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

# Amicissement du polygone
def thinPolygon(points, distance) -> list[list[float, float]]:  
    """
    #   Calcul toutes les lignes parallele aux segments du polygone et les translate de 'distance' vers l'intérieur
    #   Calcul les intersection de chaque ligne et ne garde que celle qui sont à l'intérieur du polygone
    #   Supprime tous les points qui sont à une distance inférieur à 'distance' d'un segement du polygone
    #   Renvoie l'enveloppe convexe des points restants grace à l'algorithme de Jarvis
    """

    lineCoefs = []
    for i in range(len(points)):
        a1,b1,c1 = findLineEq(points[i], points[i-1])
        (a2, b2, c2) = calculate_parallel_line(a1,b1,c1,distance)
        a3,b3,c3 = findLineEq(points[i-1], points[i-2])
        (a4, b4, c4) = calculate_parallel_line(a3,b3,c3,distance)
        if (a2, b2, c2) not in lineCoefs:
            lineCoefs.append((a2, b2, c2))
        if (a4, b4, c4) not in lineCoefs:
            lineCoefs.append((a4, b4, c4))

    temporaryPolygon = []
    for i in range(len(lineCoefs)):
        for j in range(i+1, len(lineCoefs)):
            a1, b1, c1 = lineCoefs[i]
            a2, b2, c2 = lineCoefs[j]
            p = findIntersection(a1,b1,c1,a2,b2,c2)
            if point_in_polygon(p, points):
                temporaryPolygon.append(p)
    
    thinnedPolygon = []
    for i in range(len(temporaryPolygon)):
        for j in range(len(points)):
            a,b,c = findLineEq(points[j], points[j-1])
            if orthogonalProjection(temporaryPolygon[i], a, b, c) >= distance:
                thinnedPolygon.append(temporaryPolygon[i])

    return convex_hull(thinnedPolygon)

# Renvoie l'angle entre deux objet par rapport à 0 rad
def getAbsoluteAngle(p1, p2):
    if p1[1] <= p2[1]:
        if p1[0] < p2[0]:
            ang = np.arctan((p1[1] - p2[1])/(p1[0] - p2[0]))
        elif p1[0] > p2[0]:
            ang = np.pi + np.arctan((p1[1] - p2[1])/(p1[0] - p2[0]))
        else:
            ang = np.pi / 2
    elif p1[0] > p2[0]:
        ang = np.pi + np.arctan((p1[1] - p2[1])/(p1[0] - p2[0]))
    else:
        if p1[0] == p2[0]:
            ang = 3 * np.pi / 2
        else:
            ang = 2*np.pi + np.arctan((p1[1] - p2[1])/(p1[0] - p2[0]))
    return ang 

# Fonction poid pour choisir un point où aller
def bestDirFunction(point : list[float,float], centers, boat, waypoint, parameters : list) -> list[float,float]:
    """
    #   parameters = (Vent : (direction du vent (rad), force du vent), Courrant : (direction du courant (rad), force du courant), boat : [(x, y), cap, speed], waypoint : (x, y))
    #   polygon = le polygone du diagramme de Voronoi entourant le bateau
    #   centers = coordonnées des obstacles
    """
    
    # Poids relatif à la direction relative du vent par rapport au bateau
    p1 = 1
    # Poids relatif à la force et la direction du courant
    p2 = 0
    # Poids relatif au nombre d'obstacle au alentour dans le cône de manoeuvrabilité du bateau
    p3 = 30
    # Poids relatif à l'obstacle le plus proche 
    p4 = 1
    # Poids relatif à la distance au waypoint
    p5 = 5
    # Poids relatif à l'angle que doit parcourir le voilier pour atteindre son nouveaux cap
    p6 = 1
    # Poids relatif au blocage du bateau par des obstables plus le bateau est bloqué, plus il va avoir tendance a selectionner un point loins de lui pour s'échapper et se débloquer
    p7 = 0

    wind = parameters["wind"]
    current = parameters["current"]

    cap = getAbsoluteAngle(boat.position, point)

    windDir = (wind[0] - cap)%(2*np.pi)
    if windDir >= np.pi/6 and windDir <= np.pi:
        x1 = p1/(0.246*windDir**3-1.75*windDir**2+3.73*windDir-1.5)
    elif windDir >= np.pi and windDir <= 11/6*np.pi:
        x1 = p1/(0.246*(-windDir%np.pi)**3-1.75*(-windDir%np.pi)**2+3.73*(-windDir%np.pi)-1.5)
    else:
        x1 = p1*1000
    
    currentDir = current[0]%(2*np.pi)
    if currentDir >= np.pi/6 and currentDir <= np.pi:
        x2 = p2/(0.246*currentDir**3-1.75*currentDir**2+3.73*currentDir-1.5)
    elif currentDir >= np.pi and currentDir <= 11/6*np.pi:
        x2 = p2/(0.246*(-currentDir%np.pi)**3-1.75*(-currentDir%np.pi)**2+3.73*(-currentDir%np.pi)-1.5)
    else:
        x2 = 0

    distances = [euclidean_distance(p, point) for p in centers]
    
    x3 = p3/np.sum(distances)*len(distances)

    x4 = p4/min(distances)

    x5 = p5*euclidean_distance(waypoint, point)

    ang = abs(boat.cap - getAbsoluteAngle(boat.position, point))
    x6 = p6 * (1 * (ang <= np.pi) * ang + 1*(ang > np.pi)*(2*np.pi - ang))

    d_secu = 1.5
    offset_angle = 0
    x_secu = 0
    for p in centers[1::]:
        d = euclidean_distance(boat.position, p)
        if d < d_secu and d > 1:
            angBoatObst = getAbsoluteAngle(boat.position, p)
            alpha = np.arcsin(1 / d)
            if angBoatObst - alpha < 0:
                offset_angle = 2*np.pi - (angBoatObst - alpha)%2*np.pi
            elif angBoatObst + alpha > 2*np.pi:
                offset_angle = - (angBoatObst + alpha)%2*np.pi

            if cap + offset_angle < angBoatObst + alpha + offset_angle and cap + offset_angle > angBoatObst - alpha + offset_angle:
                x_secu = 1e9
                break
        
    # with open('data.txt', 'a') as f:
    #     f.write(f"{point[0]}\t{point[1]}\t{x1}\t{x2}\t{x3}\t{x4}\t{x5}\t{x6}\t{boat.cap}\t{getAbsoluteAngle(boat.position, point)}\n")
    return x1+x2+x3+x4+x5+x6+x_secu

# Selection le point etape suivant afin d'eviter au mieux les obstacles
def selectNextPoint(thinnedPolygon, centers, boat, waypoint, parameters):
    """
    ## Description
        Selecionne et renvoie le point qui coute le moins cher pour y aller

    ### Args:
        thinnedPolygon (List): List des sommets du polygone aminci de diagramme de voronoi
        centers (List): Coordonées des centres des obstacles
        boat (Boat): Bateau
        waypoint (List): Position du waypoint
        parameters (_type_): _description_
    
    ### Returns:
        Point : Renvoie le point dont la valeur de la fonction cout est la plus faible 
    """
    # Cherche toutes les equations qui intersectent avec le cercle du champ de vision du bateau
    equations = []
    for i in range(len(thinnedPolygon)):
        p1 = thinnedPolygon[i]
        p2 = thinnedPolygon[i-1]
        if euclidean_distance(p1, boat.position) < 2 or euclidean_distance(p2, boat.position) < 2 or len(thinnedPolygon) <= 4:
            if p1 != p2:
                equations.append(findLineEq(p1, p2))
    
    # Donne la liste des points a prendre en compte pour discretiser le domaine afin de trouver le chemin avec le meilleur score
    pointsToCheck = []
    r, n_sample = 2, 72         # Resolution de 5°
    Z = np.linspace(0, 2*np.pi, n_sample)

    start = time.perf_counter()
    for z in Z:
        x, y = r*cos(z) + boat.x, r*sin(z) + boat.y
        ac, bc, cc = findLineEq(boat.position, (x,y))
        min_p = (x,y)
        for a,b,c in equations:
            point = findIntersection(a,b,c,ac,bc,cc)
            if point != None and euclidean_distance(point, boat.position) < 2 and np.sign((x-boat.x)*(point[0]-boat.x)) >= 0 and np.sign((y-boat.y)*(point[1]-boat.y)) >= 0:
                if euclidean_distance(point, boat.position) < euclidean_distance(min_p, boat.position):
                    min_p = point
        
        pointsToCheck.append(min_p)
    end = time.perf_counter()
    print(f"Point to check execution time : {end-start:.3e} s")
    
    start = time.perf_counter()
    # Recherche le point ayant le score le plus faible
    next_point = pointsToCheck[0]
    minimum_score = bestDirFunction(pointsToCheck[0], centers, boat, waypoint, parameters)
    for p in pointsToCheck[1::]:
        score = bestDirFunction(p, centers, boat, waypoint, parameters)
        if score < minimum_score:
            next_point = p
            minimum_score = score

    end = time.perf_counter()
    print(f"Select next point execution time : {end-start:.3e} s")
    
    return next_point, pointsToCheck

# Determine s'il est nécessaire de se mettre en mode evitement et renvoie la prochaine coordonée à suivre
def isAvoidanceNeeded(facets, centers, boat, waypoint : list[float, float], parameters) -> list[float, float]:
    """
    ## Description
        Determine si l'algorithme d'evitement d'obstacle est necessaire  
    
    ### Args:
        facets (list): Liste des cotes des polygones du diagramme de Voronoi
        centers (list): Positions des sommets du diagramme de Voronoi
        boat (Boat): Bateau
        waypoint (list[float, float]): Position du prochain point etape
        parameters (dict): Dictionnaire des parametres (vent, courrant, ...)
    
    ### Returns:
        list: Position du prochain point a atteindre
    """
    if isInCircle(waypoint, boat.position, 0.5) or len(centers) == 1:
        return waypoint
    
    for i in range(1, len(centers)):
        if isInCircle(centers[i], boat.position, 4): # Condition seulement dans le 'Simulateur' les objets en dehors du champ de vision sont invisibles
            start = time.perf_counter()
            thinnedPolygon = thinPolygon(facets[0], 0.5)
            end = time.perf_counter()
            print(f"Thin polygone execution time : {end-start:.3e} s")
            nextPoint = selectNextPoint(thinnedPolygon, centers[1::], boat, waypoint, parameters)
            return nextPoint
    return waypoint

def buildVoronoiDiagram(boat, obstacles):
    """
    ## Description
        Construction du diagramme de Voronoi avec les obstacles et le bateau
    
    ### Args:
        boat (Boat): Objet bateau
        obstacles (list[Obstacle]): Liste des obstacles
    
    ### Returns:
        list: Liste des centres et côtés du diagramme de Voronoi
    """
    # Definition de la zone pour effectuer la subdivision
    rect = (int(boat.x - 10), int(boat.y - 10), 20, 20)
    subdiv = cv2.Subdiv2D(rect)

    # Insertion des points dans la subdivision pour en extraire le diagram de Voronoi
    subdiv.insert(boat.position)
    for obst in obstacles:
        if isInCircle(obst.position, boat.position, 4):
            subdiv.insert(obst.position)

    # Calcul du diagramme de Voronoi
    (facets, centers) = subdiv.getVoronoiFacetList([])

    return facets, centers
