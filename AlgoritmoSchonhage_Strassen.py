import numpy as np
import time
def schoenhage_multiply(x, y):
    # Convierte los números en listas de dígitos en base 10
    x_digits = [int(digit) for digit in str(x)]
    y_digits = [int(digit) for digit in str(y)]
    
    # Calcula el tamaño necesario para la convolución
    size = 2**np.ceil(np.log2(len(x_digits) + len(y_digits) - 1)).astype(int)
    
    # Realiza la convolución mediante la FFT
    x_fft = np.fft.fft(x_digits, size)
    y_fft = np.fft.fft(y_digits, size)
    z_fft = x_fft * y_fft
    
    # Obtiene los dígitos del producto mediante la inversa de la FFT
    z_digits = np.round(np.fft.ifft(z_fft)).real.astype(int)
    # Realiza el acarreo de los dígitos
    carry = 0
    result = []
    for digit in reversed(z_digits):
        if digit != 0:
            value = digit + carry
            result.insert(0, str(value % 10))
            carry = value // 10
    
    # Combina los dígitos en el producto final
    if carry > 0:
        result.insert(0, str(carry))

    num = int(''.join(result))
    return num

# Programa principal #
x = 123456789123
y = 987654321987
# Solucion
timeIni = time.time()
product = schoenhage_multiply(x, y)
timeFin = time.time() - timeIni
print(product)
print("\n *Tiempo de ejecucion:", timeFin, "*")
