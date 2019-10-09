#ICS435
#Nathan Onaka
#Soft K-Means

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

#imports points from file
data = pd.read_csv('data.txt', header=None)

#matplotlib plot function
def plot(points, distortion, K, centers):
    random_colors = np.random.random((K, 3))
    #takes dot product of the distortion level * the random color array to depict the graph
    colors = distortion.dot(random_colors)

    #prints all distortion values for every point for analysis
    #print(distortion)

    #prints average distortion distribution for each point
    first = distortion[:40]
    print(np.average(first, axis=0))
    second = distortion[40:80]
    print(np.average(second, axis=0))
    third = distortion[80:120]
    print(np.average(third, axis=0))

    plt.scatter(points[:, 0], points[:, 1], c=colors)
    plt.scatter(centers[:, 0], centers[:, 1], marker ='*', c='#050505', s=100)
    plt.show()

#sets initial center positions
def initialCenters():
    #centers were assigned randomly at first until more optimal centers were found.
    #centers = np.array([[random.randint(-20, 20),random.randint(-20, 20)],[random.randint(-20, 20),random.randint(-20, 20)],[random.randint(-20, 20),random.randint(20, 20)]])
    centers = np.array([[0, -20], [10, -10], [20, 0]])
    return centers

#updates centers position
def updateCenters(points, distortion, K):
    row, col = points.shape
    centers = np.zeros((K, col))
    for i in range(K):
        centers[i] = distortion[:, i].dot(points) / distortion[:, i].sum()
    return centers

#calculate distortion of each point using beta function
def betaFormula(centers, points, beta):
    row, _ = points.shape
    cenRow, cenCol = centers.shape
    distortion = np.zeros((row, cenRow))

    for i in range(row):
        #distortion formula, found in lecture notes and online resources
        distortion[i] = np.exp(-beta * np.linalg.norm(centers - points[i], 2, axis=1))
    distortion /= distortion.sum(axis=1, keepdims=True)
    return distortion

#softKMeans function, sets iterations, beta value, and number of centers
def softKMeans(points, K, iters=30, beta=.4):
    centers = initialCenters()
    for _ in range(iters):
        distortion = betaFormula(centers, points, beta)
        centers = updateCenters(points, distortion, K)
    plot(points, distortion, K, centers)
    print(centers)

#creates an array of the imported points, calls the softKMeans function
x = np.array(data)
softKMeans(x, K=3)

