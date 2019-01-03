import numpy as np

def df_dx(x, y, z, points):
    result = 0
    for i in range(0, len(points)):
        result = result + (x - points[i][0]) / np.sqrt((x - points[i][0])**2 + (y - points[i][1])**2 + (z - points[i][2])**2)
    result = (1 / len(points)) * result
    return result

def df_dy(x, y, z, points):
    result = 0
    for i in range(0, len(points)):
        result = result + (y - points[i][1]) / np.sqrt((x - points[i][0])**2 + (y - points[i][1])**2 + (z - points[i][2])**2)
    result = (1 / len(points)) * result
    return result
    
def df_dz(x, y, z, points):
    result = 0
    for i in range(0, len(points)):
        result = result + (z - points[i][2]) / np.sqrt((x - points[i][0])**2 + (y - points[i][1])**2 + (z - points[i][2])**2)
    result = (1 / len(points)) * result
    return result

def d2f_dx2(x, y, z, points):
    result = 0
    for i in range(0, len(points)):
        result = result + ((y - points[i][1])**2 + (z - points[i][2])**2) / ((x - points[i][0])**2 + (y - points[i][1])**2 + (z - points[i][2])**2)**(3/2)
    result = (1 / len(points)) * result
    return result
    
def d2f_dxdy(x, y, z, points):
    result = 0
    for i in range(0, len(points)):
        result = result + (x - points[i][0]) * (y - points[i][1]) / ((x - points[i][0])**2 + (y - points[i][1])**2 + (z - points[i][2])**2)**(3/2)
    result = (-1 / len(points)) * result
    return result

def d2f_dxdz(x, y, z, points):
    result = 0
    for i in range(0, len(points)):
        result = result + (x - points[i][0]) * (z - points[i][2]) / ((x - points[i][0])**2 + (y - points[i][1])**2 + (z - points[i][2])**2)**(3/2)
    result = (-1 / len(points)) * result
    return result

def d2f_dydx(x, y, z, points):
    result = 0
    for i in range(0, len(points)):
        result = result + (y - points[i][1]) * (x - points[i][0]) / ((x - points[i][0])**2 + (y - points[i][1])**2 + (z - points[i][2])**2)**(3/2)
    result = (-1 / len(points)) * result
    return result
        
def d2f_dy2(x, y, z, points):
    result = 0
    for i in range(0, len(points)):
        result = result + ((x - points[i][0])**2 + (z - points[i][2])**2) / ((x - points[i][0])**2 + (y - points[i][1])**2 + (z - points[i][2])**2)**(3/2)
    result = (1 / len(points)) * result
    return result

def d2f_dydz(x, y, z, points):
    result = 0
    for i in range(0, len(points)):
        result = result + (y - points[i][1]) * (z - points[i][2]) / ((x - points[i][0])**2 + (y - points[i][1])**2 + (z - points[i][2])**2)**(3/2)
    result = (-1 / len(points)) * result
    return result

def d2f_dzdx(x, y, z, points):
    result = 0
    for i in range(0, len(points)):
        result = result + (z - points[i][2]) * (x - points[i][0]) / ((x - points[i][0])**2 + (y - points[i][1])**2 + (z - points[i][2])**2)**(3/2)
    result = (-1 / len(points)) * result
    return result

def d2f_dzdy(x, y, z, points):
    result = 0
    for i in range(0, len(points)):
        result = result + (z - points[i][2]) * (y - points[i][1]) / ((x - points[i][0])**2 + (y - points[i][1])**2 + (z - points[i][2])**2)**(3/2)
    result = (-1 / len(points)) * result
    return result

def d2f_dz2(x, y, z, points):
    result = 0
    for i in range(0, len(points)):
        result = result + ((x - points[i][0])**2 + (y - points[i][1])**2) / ((x - points[i][0])**2 + (y - points[i][1])**2 + (z - points[i][2])**2)**(3/2)
    result = (1 / len(points)) * result
    return result

def inv_22(matrix):
    if (matrix[0][0]*matrix[1][1] - matrix[0][1] * matrix[1][0] == 0):
        raise Exception("There is no inverse matrix.")
    else:
        d = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        return [[matrix[1][1] / d, 0 - matrix[0][1] / d], [0 - matrix[1][0] / d, matrix[0][0] / d]]

def inv_33(matrix):
    m1 = [[matrix[1][1], matrix[1][2]], [matrix[2][1], matrix[2][2]]]
    m2 = [[matrix[1][0], matrix[1][2]], [matrix[2][0], matrix[2][2]]]
    m3 = [[matrix[1][0], matrix[1][1]], [matrix[2][0], matrix[2][1]]]
    d = matrix[0][0] * (m1[0][0] * m1[1][1] - m1[0][1] * m1[1][0]) - matrix[0][1] * (m2[0][0] * m2[1][1] - m2[0][1] * m2[1][0]) + matrix[0][2] * (m3[0][0] * m3[1][1] - m3[0][1] * m3[1][0])
    if (d == 0):
        raise Exception("There is no inverse matrix for this 3 by 3.")
    else:
        #Transpose the original matrix
        matrix = [[matrix[0][0], matrix[1][0], matrix[2][0]], [matrix[0][1], matrix[1][1], matrix[2][1]], [matrix[0][2], matrix[1][2], matrix[2][2]]]
        #Find the determinant of the that uncrossed out i-th row and j-th column
        a11 = matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]
        a12 = matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]
        a13 = matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]
        a21 = matrix[0][1] * matrix[2][2] - matrix[0][2] * matrix[2][1]
        a22 = matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]
        a23 = matrix[0][0] * matrix[2][1] - matrix[0][1] * matrix[2][0]
        a31 = matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]
        a32 = matrix[0][0] * matrix[1][2] - matrix[0][2] * matrix[1][0]
        a33 = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        #Multiply the transposed matrix by the cofactors
        a12 = a12 * -1
        a21 = a21 * -1
        a23 = a23 * -1
        a32 = a32 * -1
        #Divide the resulting adjacent matrix by the determinant
        matrix = [[a11 / d, a12 / d, a13 / d], [a21 / d, a22 / d, a23 / d], [a31 / d, a32 / d, a33 / d]]
        return matrix

def Newt_Raph(p0, q0, r0, maxI, eps, points):
    d = 256
    k = 0
    p_k = [[p0], [q0], [r0]]
    while (k < maxI and d > eps):
        print("K:",k,"; X:", p_k[0][0],"; Y:", p_k[1][0], "; Z:", p_k[2][0])
        J_k = [[d2f_dx2(p_k[0][0], p_k[1][0], p_k[2][0], points), d2f_dxdy(p_k[0][0], p_k[1][0], p_k[2][0], points), d2f_dxdz(p_k[0][0], p_k[1][0], p_k[2][0], points)], 
               [d2f_dydx(p_k[0][0], p_k[1][0], p_k[2][0], points), d2f_dy2(p_k[0][0], p_k[1][0], p_k[2][0], points), d2f_dydz(p_k[0][0], p_k[1][0], p_k[2][0], points)], 
               [d2f_dzdx(p_k[0][0], p_k[1][0], p_k[2][0], points), d2f_dzdy(p_k[0][0], p_k[1][0], p_k[2][0], points), d2f_dz2(p_k[0][0], p_k[1][0], p_k[2][0], points)]]
        IJ_k = inv_33(J_k)
        F_k = [[df_dx(p_k[0][0], p_k[1][0], p_k[2][0], points)], 
               [df_dy(p_k[0][0], p_k[1][0], p_k[2][0], points)], 
               [df_dz(p_k[0][0], p_k[1][0], p_k[2][0], points)]]
        p_i = [[p_k[0][0] - (IJ_k[0][0] * F_k[0][0] + IJ_k[0][1] * F_k[1][0] + IJ_k[0][2] * F_k[2][0])], 
               [p_k[1][0] - (IJ_k[1][0] * F_k[0][0] + IJ_k[1][1] * F_k[1][0] + IJ_k[1][2] * F_k[2][0])], 
               [p_k[2][0] - (IJ_k[2][0] * F_k[0][0] + IJ_k[2][1] * F_k[1][0] + IJ_k[2][2] * F_k[2][0])]]
        d = np.sqrt((p_i[0][0] - p_k[0][0])**2 + (p_i[1][0] - p_k[1][0])**2 + (p_i[2][0] - p_k[2][0])**2)
        k = k + 1
        p_k = [[p_i[0][0]], [p_i[1][0]], [p_i[2][0]]]
    return p_k
    
points = [[1, 1, 1],[-1, 2, 1],[3, 4, 1],[5, -4, 1],[-2, -4, 1],[3, 8, 1],[6, 2, 1]]
x = 0
y = 0
z = 0
for i in range(0, len(points)):
    x = x + points[i][0]
    y = y + points[i][1]
    z = z + points[i][2]
x = x / len(points)
y = y / len(points)
z = z / len(points)
p = Newt_Raph(x, y, z, 250, 1e-11, points)
print("[[", p[0][0], "],[", p[1][0], "],[", p[2][0], "]]")
