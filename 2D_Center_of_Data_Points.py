import numpy as np

def df_dx(x,y, points):
    result = 0
    for i in range(0, len(points)):
        result = result + (x - points[i][0]) / np.sqrt((x - points[i][0])**2 + (y - points[i][1])**2)
    result = (1 / len(points)) * result
    return result

def df_dy(x,y, points):
    result = 0
    for i in range(0, len(points)):
        result = result + (y - points[i][1]) / np.sqrt((x - points[i][0])**2 + (y - points[i][1])**2)
    result = (1 / len(points)) * result
    return result
    
def d2f_dx2(x,y, points):
    result = 0
    for i in range(0, len(points)):
        result = result + (y - points[i][1])**2 / ((x - points[i][0])**2 + (y - points[i][1])**2)**(3/2)
    result = (1 / len(points)) * result
    return result
    
def d2f_dxdy(x,y, points):
    result = 0
    for i in range(0, len(points)):
        result = result + (x - points[i][0]) * (y - points[i][1]) / ((x - points[i][0])**2 + (y - points[i][1])**2)**(3/2)
    result = (-1 / len(points)) * result
    return result
    
def d2f_dydx(x,y, points):
    result = 0
    for i in range(0, len(points)):
        result = result + (y - points[i][1]) * (x - points[i][0]) / ((x - points[i][0])**2 + (y - points[i][1])**2)**(3/2)
    result = (-1 / len(points)) * result
    return result
        
def d2f_dy2(x,y, points):
    result = 0
    for i in range(0, len(points)):
        result = result + (x - points[i][0])**2 / ((x - points[i][0])**2 + (y - points[i][1])**2)**(3/2)
    result = (1 / len(points)) * result
    return result
    
def inv_22(matrix):
    if (matrix[0][0]*matrix[1][1] - matrix[0][1] * matrix[1][0] < 0):
        raise Exception("There is no inverse matrix.")
    else:
        d = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        return [[matrix[1][1] / d, 0 - matrix[0][1] / d], [0 - matrix[1][0] / d, matrix[0][0] / d]]

def Newt_Raph(p0, q0, maxI, eps, points):
    d = 256
    k = 0
    p_k = [[p0], [q0]]
    while (k < maxI and d > eps):
        print("K:",k,"; X:", p_k[0][0],"; Y:", p_k[1][0])
        J_k = [[d2f_dx2(p_k[0][0], p_k[1][0], points), d2f_dxdy(p_k[0][0], p_k[1][0], points)], [d2f_dydx(p_k[0][0], p_k[1][0], points), d2f_dy2(p_k[0][0], p_k[1][0], points)]]
        IJ_k = inv_22(J_k)
        F_k = [[df_dx(p_k[0][0], p_k[1][0], points)], [df_dy(p_k[0][0], p_k[1][0], points)]]
        p_i = [[p_k[0][0] - (IJ_k[0][0] * F_k[0][0] + IJ_k[0][1] * F_k[1][0])], [p_k[1][0] - (IJ_k[1][0] * F_k[0][0] + IJ_k[1][1] * F_k[1][0])]]
        d = np.sqrt((p_i[0][0] - p_k[0][0])**2 + (p_i[1][0] - p_k[1][0])**2)
        k = k + 1
        p_k = [[p_i[0][0]], [p_i[1][0]]]
    return p_k
    
points = [[1, 1],[-1, 2],[3, 4],[5, -4],[-2, -4],[3, 8],[6, 2]]
x = 0
y = 0
for i in range(0, len(points)):
    x = x + points[i][0]
    y = y + points[i][1]
x = x / len(points)
y = y / len(points)
p = Newt_Raph(x, y, 250, 1e-11, points)
print("[[", p[0][0], "],[", p[1][0], "]]")
