#
import numpy as np
import random as rand
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

X = 1
Y = 1
Z = 1
dt = 0.001

def pos_Cal(t, v):
    return t + v * dt;

def position(x, y, z, v1, v2, v3):
    vx = v1
    vy = v2
    vz = v3
    return [pos_Cal(x, vx), pos_Cal(y, vy), pos_Cal(z, vz)]

def randSlightGrid():
    dx = 0.01
    dy = 0.01
    dz = 0.01
    grid = np.zeros((3, 10000))
    for t in range(0, int(Y / dy)):
        for v in range(0, int(X / dx)):
            mu = .05
            grid[0][100 * t + v] = t * dx
            grid[1][100 * t + v] = v * dy
            if (t != 0 and v != 0):
                mu = (grid[2][100 * (t - 1) + v - 1] + grid[2][100 * (t - 1) + v] + grid[2][100 * (t - 1) + v + 1] + grid[2][100 * t + v - 1]) / 4.0
            elif (t != 0):
                mu = (grid[2][100 * (t - 1) + v] + grid[2][100 * (t - 1) + v + 1]) / 4.0
            b = 3.0 * dz * np.sin(2.0 * np.pi * rand.random())
            grid[2][100 * t + v] = abs(mu + b)
#    for t in range(0, int(Y / dy)):
#        for v in range(0, int(X / dx)):
#            mu = .05
#            grid[0][100 * v + t] = t * dx
#            grid[1][100 * v + t] = v * dy
#            if (t != 0 and v != 0):
#                mu = (grid[2][100 * (t - 1) + v - 1] + grid[2][100 * (t - 1) + v] + grid[2][100 * (t - 1) + v + 1] + grid[2][100 * t + v - 1]) / 4.0
#            elif (t != 0):
#                mu = (grid[2][100 * (t - 1) + v] + grid[2][100 * (t - 1) + v + 1]) / 4.0
#            b = 3.0 * dz * np.sin(2.0 * np.pi * rand.uniform(0,1))
#            grid[2][100 * t + v] = mu + b
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_wireframe(grid[0][:], grid[1][:], grid[2][:])
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.01)
    ax.set_zlim(0, 1.01)
    plt.show()
    
def randRoughGrid():
    dx = 0.01
    dy = 0.01
    grid = np.zeros((3, 10000))
    for t in range(0, int(Y / dy)):
        for v in range(0, int(X / dx)):
            grid[0][100 * t + v] = t * dx
            grid[1][100 * t + v] = v * dy
            grid[2][100 * t + v] = .2
    for t in range(0, 30000):
        a = rand.randint(0, 99)
        b = rand.randint(0, 99)

        if (a == 0 and b == 0):
            mu = (grid[2][9998] + grid[2][9999] + grid[2][9900] + grid[2][99] + grid[2][1] +  grid[2][199] + grid[2][100] + grid[2][101]) / 8.0
            sigma = np.abs(np.median([grid[2][0] - grid[2][9998],grid[2][0] - grid[2][9999],grid[2][0] - grid[2][9900],grid[2][0] - grid[2][99],grid[2][0] - grid[2][1],grid[2][0] - grid[2][199],grid[2][0] - grid[2][100],grid[2][0] - grid[2][101]])) / 8.0
        elif (a == 99 and b == 99):
            mu = (grid[2][9898] + grid[2][9899] + grid[2][9800] + grid[2][9998] + grid[2][9900] +  grid[2][198] + grid[2][199] + grid[2][0]) / 8.0
            sigma = np.abs(np.median([grid[2][9999] - grid[2][9898],grid[2][9999] - grid[2][9899],grid[2][9999] - grid[2][9800],grid[2][9999] - grid[2][9998],grid[2][9999] - grid[2][9900],grid[2][9999] - grid[2][198],grid[2][9999] - grid[2][199],grid[2][9999] - grid[2][0]])) / 8.0
        elif (a == 0 and b == 99):
            mu = (grid[2][9998] + grid[2][9999] + grid[2][9900] + grid[2][98] + grid[2][1] +  grid[2][198] + grid[2][199] + grid[2][100]) / 8.0
            sigma = np.abs(np.median([grid[2][99] - grid[2][9998],grid[2][99] - grid[2][9999],grid[2][99] - grid[2][9900],grid[2][99] - grid[2][98],grid[2][99] - grid[2][1],grid[2][99] - grid[2][198],grid[2][99] - grid[2][199],grid[2][99] - grid[2][100]])) / 8.0
        elif (a == 99 and b == 0):
            mu = (grid[2][9899] + grid[2][9800] + grid[2][9801] + grid[2][9999] + grid[2][9901] +  grid[2][99] + grid[2][0] + grid[2][1]) / 8.0
            sigma = np.abs(np.median([grid[2][9900] - grid[2][9899],grid[2][9900] - grid[2][9800],grid[2][9900] - grid[2][9801],grid[2][9900] - grid[2][9999],grid[2][9900] - grid[2][9901],grid[2][9900] - grid[2][99],grid[2][9900] - grid[2][0],grid[2][9900] - grid[2][1]])) / 8.0
        elif (a == 0):
            mu = (grid[2][9900+(b-1)] + grid[2][9900+b] + grid[2][9900+(b+1)] + grid[2][b-1] + grid[2][b+1] +  grid[2][100+(b-1)] + grid[2][100+b] + grid[2][100+(b+1)]) / 8.0
            sigma = np.abs(np.median([grid[2][b] - grid[2][9900+(b-1)],grid[2][b] - grid[2][9900+b],grid[2][b] - grid[2][9900+(b+1)],grid[2][b] - grid[2][b-1],grid[2][b] - grid[2][b+1],grid[2][b] - grid[2][100+(b-1)],grid[2][b] - grid[2][100+b],grid[2][b] - grid[2][100+(b+1)]])) /8.0
        elif (b == 0):
            mu = (grid[2][100*(a-1)+99] + grid[2][100*(a-1)] + grid[2][100*(a-1)+(b+1)] + grid[2][100*a+99] + grid[2][100*a+(b+1)] +  grid[2][100*(a+1)+99] + grid[2][100*(a+1)] + grid[2][100*(a+1)+(b+1)]) / 8.0
            sigma = np.abs(np.median([grid[2][100*a] - grid[2][100*(a-1)+99],grid[2][100*a] - grid[2][100*(a-1)],grid[2][100*a] - grid[2][100*(a-1)+(b+1)],grid[2][100*a] - grid[2][100*a+99],grid[2][100*a] - grid[2][100*a+(b+1)],grid[2][100*a] - grid[2][100*(a+1)+99],grid[2][100*a] - grid[2][100*(a+1)],grid[2][100*a] - grid[2][100*(a+1)+(b+1)]])) / 8.0
        elif (a == 99):
            mu = (grid[2][9800+(b-1)] + grid[2][9800+b] + grid[2][9800+(b+1)] + grid[2][9900+(b-1)] + grid[2][9900+(b+1)] +  grid[2][b-1] + grid[2][b] + grid[2][b+1]) / 8.0
            sigma = np.abs(np.median([grid[2][9900+b] - grid[2][9800+(b-1)],grid[2][9900+b] - grid[2][9800+b],grid[2][9900+b] - grid[2][9800+(b+1)],grid[2][9900+b] - grid[2][9900+(b-1)],grid[2][9900+b] - grid[2][9900+(b+1)],grid[2][9900+b] - grid[2][b-1],grid[2][9900+b] - grid[2][b],grid[2][9900+b] - grid[2][b+1]])) / 8.0
        elif (b == 99):
            mu = (grid[2][100*(a-1)+(b-1)] + grid[2][100*(a-1)+b] + grid[2][100*(a-1)] + grid[2][100*a+(b-1)] + grid[2][100*a] +  grid[2][100*(a+1)+(b-1)] + grid[2][100*(a+1)+b] + grid[2][100*(a+1)]) / 8.0
            sigma = np.abs(np.median([grid[2][100*a+b] - grid[2][100*(a-1)+(b-1)],grid[2][100*a+b] - grid[2][100*(a-1)+b],grid[2][100*a+b] - grid[2][100*(a-1)],grid[2][100*a+b] - grid[2][100*a+(b-1)],grid[2][100*a+b] - grid[2][100*a],grid[2][100*a+b] - grid[2][100*(a+1)+(b-1)],grid[2][100*a+b] - grid[2][100*(a+1)+b],grid[2][100*a+b] - grid[2][100*(a+1)]])) / 8.0
        else:
            mu = (grid[2][100*(a-1)+(b-1)] + grid[2][100*(a-1)+b] + grid[2][100*(a-1)+(b+1)] + grid[2][100*a+(b-1)] + grid[2][100*a+(b+1)] + grid[2][100*(a+1)+(b-1)] + grid[2][100*(a+1)+b] + grid[2][100*(a+1)+(b+1)]) / 8.0
            sigma = np.abs(np.median([grid[2][100*a+b] - grid[2][100*(a-1)+(b-1)],grid[2][100*a+b] - grid[2][100*(a-1)+b],grid[2][100*a+b] - grid[2][100*(a-1)+(b+1)],grid[2][100*a+b] - grid[2][100*a+(b-1)],grid[2][100*a+b] - grid[2][100*a+(b+1)],grid[2][100*a+b] - grid[2][100*(a+1)+(b-1)],grid[2][100*a+b] - grid[2][100*(a+1)+b],grid[2][100*a+b] - grid[2][100*(a+1)+(b+1)]])) / 8.0
        grid[2][100 * a + b] = mu + (-1.0 + 2.0 * rand.random()) * rand.gauss(mu, sigma)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.scatter(grid[0][:], grid[1][:], grid[2][:])
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.01)
    ax.set_zlim(0, 1.01)
    plt.show()
    
#randSlightGrid()
randRoughGrid()
