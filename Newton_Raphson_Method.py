import numpy as np

def f1(x, y):
    return np.sqrt((x - 1)**2+(y - 1)**2)

def f2(x, y):
    return np.sqrt((x - 3)**2+(y - 8)**2)

def f3(x, y):
    return np.sqrt((x + 1)**2+(y - 2)**2)

def f4(x, y):
    return np.sqrt((x + 2)**2+(y + 4)**2)

def f5(x, y):
    return np.sqrt((x - 5)**2+(y + 4)**2)

def f6(x, y):
    return np.sqrt((x - 3)**2+(y - 4)**2)

def f7(x, y):
    return np.sqrt((x - 6)**2+(y - 2)**2)

def f(x, y):
    return (1/7)*(f1(x,y) + f2(x,y) + f3(x,y) + f4(x,y) + f5(x,y) + f6(x,y) + f7(x,y))

def df_dx(x,y):
    return (1/7) * ((x - 1) / f1(x,y) + (x - 3) / f2(x,y) + (x + 1) / f3(x,y) + (x + 2) / f4(x,y) + (x - 5) / f5(x,y) + (x - 3) / f6(x,y) + (x - 6) / f7(x,y))

def df_dy(x,y):
    return (1/7) * ((y - 1) / f1(x,y) + (y - 8) / f2(x,y) + (y - 2) / f3(x,y) + (y + 4) / f4(x,y) + (y + 4) / f5(x,y) + (y - 4) / f6(x,y) + (y - 2) / f7(x,y))

def d2f_dx2(x,y):
    return (1/7) * ((y - 1)**2 / (f1(x,y))**3 + (y - 8)**2 / (f2(x,y))**3 + (y - 2)**2 / (f3(x,y))**3 + (y + 4)**2 / (f4(x,y))**3 + (y + 4)**2 / (f5(x,y))**3 + (y - 4)**2 / (f6(x,y))**3 + (y - 2)**2 / (f7(x,y))**3)

def d2f_dxdy(x,y):
    return (-1/7) * ((y - 1)*(x - 1) / (f1(x,y))**3 + (y - 8)*(x - 3) / (f2(x,y))**3 + (y - 2)*(x + 1) / (f3(x,y))**3 + (y + 4)*(x + 2) / (f4(x,y))**3 + (y + 4)*(x - 5) / (f5(x,y))**3 + (y - 4)*(x - 3) / (f6(x,y))**3 + (y - 2)*(x - 6) / (f7(x,y))**3)

def d2f_dydx(x,y):
    return (-1/7) * ((y - 1)*(x - 1) / (f1(x,y))**3 + (y - 8)*(x - 3) / (f2(x,y))**3 + (y - 2)*(x + 1) / (f3(x,y))**3 + (y + 4)*(x + 2) / (f4(x,y))**3 + (y + 4)*(x - 5) / (f5(x,y))**3 + (y - 4)*(x - 3) / (f6(x,y))**3 + (y - 2)*(x - 6) / (f7(x,y))**3)
    
def d2f_dy2(x,y):
    return (1/7) * ((x - 1)**2 / (f1(x,y))**3 + (x - 3)**2 / (f2(x,y))**3 + (x + 1)**2 / (f3(x,y))**3 + (x + 2)**2 / (f4(x,y))**3 + (x - 5)**2 / (f5(x,y))**3 + (x - 3)**2 / (f6(x,y))**3 + (x - 6)**2 / (f7(x,y))**3)

def inv_22(matrix):
    if (matrix[0][0]*matrix[1][1] - matrix[0][1] * matrix[1][0] < 0):
        raise Exception("There is no inverse matrix.")
    else:
        d = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        return [[matrix[1][1] / d, 0 - matrix[0][1] / d], [0 - matrix[1][0] / d, matrix[0][0] / d]]

def Newt_Raph(p0, q0, maxI, eps):
    d = 256
    k = 0
    p_k = [[p0], [q0]]
    while (k < maxI and d > eps):
        print("K:",k,"; X:", p_k[0][0],"; Y:", p_k[1][0])
        J_k = [[d2f_dx2(p_k[0][0], p_k[1][0]), d2f_dxdy(p_k[0][0], p_k[1][0])], [d2f_dydx(p_k[0][0], p_k[1][0]), d2f_dy2(p_k[0][0], p_k[1][0])]]
        IJ_k = inv_22(J_k)
        F_k = [[df_dx(p_k[0][0], p_k[1][0])], [df_dy(p_k[0][0], p_k[1][0])]]
        p_i = [[p_k[0][0] - (IJ_k[0][0] * F_k[0][0] + IJ_k[0][1] * F_k[1][0])], [p_k[1][0] - (IJ_k[1][0] * F_k[0][0] + IJ_k[1][1] * F_k[1][0])]]
        d = np.sqrt((p_i[0][0] - p_k[0][0])**2 + (p_i[1][0] - p_k[1][0])**2)
        k = k + 1
        p_k = [[p_i[0][0]], [p_i[1][0]]]
    return p_k
    
p = Newt_Raph(2, 2, 250, 1e-11)
print("[[", p[0][0], "],[", p[1][0], "]]")
