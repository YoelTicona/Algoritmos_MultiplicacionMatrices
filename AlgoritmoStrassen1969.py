import numpy as np
import time
# * Algoritmo de Volker Strassen * #
def strassen_multiply(A, B):
    n = A.shape[0]
    # Verificar si las matrices son de tamaño 1x1
    if n == 1:
        return A * B

    # Dividir las matrices en submatrices más pequeñas
    mid = n // 2
    A11 = A[:mid, :mid]
    A12 = A[:mid, mid:]
    A21 = A[mid:, :mid]
    A22 = A[mid:, mid:]

    B11 = B[:mid, :mid]
    B12 = B[:mid, mid:]
    B21 = B[mid:, :mid]
    B22 = B[mid:, mid:]

    # Calcular los siete productos de Strassen
    M1 = strassen_multiply(A21 + A22 - A11, B22 - B12 + B11)
    M2 = strassen_multiply(A11, B11)
    M3 = strassen_multiply(A12, B21)
    M4 = strassen_multiply(A11 - A21, B22 - B12)
    M5 = strassen_multiply(A21 + A22, B12 - B11)
    M6 = strassen_multiply(A12 - A21 + A11 - A22, B22)
    M7 = strassen_multiply(A22, B11 + B22 - B12 - B21)
    # Calcular las submatrices del resultado final
    C11 = M2 + M3
    C12 = M1 + M2 + M5 + M6
    C21 = M1 + M2 + M4 - M7
    C22 = M1 + M2 + M4 + M5

    # Combinar las submatrices en la matriz de resultado
    result = np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))
    return result


# ===== Programa Principal ===== #
# Matriz 1
f1 = 16
c1 = 16
A =np.array([[  6,  1, -2, 1, 10, -2,  2, -1,  3,  4,  6,  4, -2,  3,  6,  4],
             [ -2, 1, -2,  1,  2,  4,  7, -2,  7,  0,  1,  8,  7, 10,  9,  7],
             [  9, 4,  4,  1,  2,  8,  9,  8,  8, 10, 10,  6,  0, -2, 10, -2],
             [ 10, 7,  0,  8,  2,  1,  2,  3, -2,  4,  4,  1,  2, 10,  8,  8],
             [ -1, 3,  7,  1,  9,  4,  9,  3,  0,  9,  7, -2,  3, -1,  4,  9],
             [  4, 3,  9,  10, 2,  1,  6,  1, 10,  9,  5,  2,  1, -2,  1,  9],
             [ -2, 0,  1,  8,  7,  9,  5,  7,  9, -2,  9,  4,  7,  0, 10, 10],
             [  7, 8, 10,  0,  3, -2, 10,  8,  8,  3,  3,  6, -1,  1,  9, -2],
             [  7, 2, -1,  3,  7,  1,  4,  8,  1,  9,  5, -2,  9,  3,  3, -1],
             [  5, 0,  8,  2,  3,  3,  1,  6,  3,  6, -1,  3,  1, -2,  8,  9],
             [  7, 1, -1, -2,  5,  2, 10,  6, -1,  0,  2, -1,  4,  9, -2, -2],
             [  4, 7, -2, -2,  5,  4, 10,  2, -2,  9,  2,  5,  9,  7,  7, 10],
             [  7, 8, -1,  8,  4,  5,  6,  5,  9,  3,  3,  4, -2,  5,  0,  3],
             [  5, 5, -1, -1, -2,  5,  3, -2,  9,  0,  9, -2,  1,  2,  5, -2],
             [  9, 0,  3,  9,  8,  3,  5, -2,  7,  8,  4,  8,  6,  5,  8,  6],
             [  1, 5,  0,  3,  7, -2, -1,  4,  1, 10,  2,  8, -2,  6,  7, -2]])
# Matriz 2
f2 = 16
c2 = 16
B =np.array([[ 0,  2,  0,  5,  7,  6,  0,  5,  0,  1,  6,  2,  0,  5, -1,  7],
             [-1,  2,  1,  1,  1,  2,  0,  5,  6,  5,  0, -1, -2,  7,  5, -1],
             [ 2, -2,  1,  7,  6, -2, -1,  7,  4, -2,  3,  0,  6, -2, -2,  6],
             [ 2,  3,  4,  3,  1,  6,  4,  5,  6,  6,  0,  7,  1, -2, -2,  6],
             [ 4,  4, -2,  0, -1, -1,  6,  7,  6,  1,  4,  7, -1, -2,  3,  4],
             [ 7,  4,  3,  2,  2,  7,  7,  3,  7,  5,  1,  4,  5,  3, -2, -1],
             [ 4, -2,  3,  0, -1,  2,  1,  2,  0,  5,  6,  4, -1,  5, -2,  5],
             [ 4,  0,  1,  3, -1,  5, -2,  6,  4,  0,  7,  3,  7,  1,  6,  5],
             [ 4,  5,  5, -1, -2,  1,  2,  0,  1,  2,  0,  5,  7,  4,  5,  0],
             [-2,  7,  0, -1,  5,  5,  1,  0,  0, -1,  7,  7,  5,  6,  2, -1],
             [ 0, -2,  5,  3,  7, -2, -2,  4,  4,  6, -1,  7, -1, -2, -1,  2],
             [ 2,  3,  2,  5,  4,  1,  3,  4,  1,  4,  3,  1,  7,  6,  2,  2],
             [ 3,  7,  1, -1,  6,  6, -1, -2, -1,  2,  0,  1,  4,  3,  4, -2],
             [ 5,  4,  4,  4,  3,  4,  5,  5, -2, -1,  5,  4, -2,  0,  3, -1],
             [ 0,  5, -2,  4,  0,  7,  4,  4,  3, -2,  5,  5,  2,  0, -2,  7],
             [ 2,  2,  7,  4, -1,  0,  2,  3,  0,  4,  7, -1,  7, -2,  6,  6]])

# Solucion
tiempoIni = time.time()
C = strassen_multiply(A, B)

# Imprimiendo
print("* Matriz A *")
for fila in A:
    print(fila)
print("* Matriz B *")
for fila in B:
    print(fila)
print("* Matriz C *")
for fila in C:
    print(fila)
tiempoFin = time.time() - tiempoIni
print("\n*Tiempo de ejecucion", tiempoFin, " *")
