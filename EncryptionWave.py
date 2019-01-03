# Encrypting a string of words into 256-bit
import numpy as np
import pylab

total_Length = 37
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
            'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', 
            '6', '7', '8', '9', ' '];

def changeLetter(length, letter, index):
    return (1.0 * index + 1.0 * letter / total_Length) / length

def wave(x, values):
    for y in values:
        if x == y:
            return np.sin(2 * np.pi * x) + np.sign(np.sin(2 * np.pi * x)) * 0.5
    return np.sin(2 * np.pi * x)

def Encrypt(line):
    line = line.lower()
    values = [0] * len(line)
    for j in range(0, len(line)):
        for i in range(0, len(letters)):
            if line[j] == letters[i]:
                values[j] = i + 1
    length = len(line);
    values = [round(changeLetter(length, values[i], i), 4) for i in range(0, len(values))]
    sinal = [round(wave(1.0 * i / 10000.0, values), 4) for i in range(0, 10000)]
    return sinal

def Decrypt(line):
    real = [round(np.sin(2 * np.pi * x / 10000), 4) for x in range(0, 10000)]
    sub = [line[i] - real[i] for i in range(0, 10000)]
    values = [1.0 * i / 10000 for i in range(0, len(sub)) if (sub[i] != 0.0)]
    result = ""
    if (len(values) > 270):
        raise Exception("Not a respected wavelength")
    print(values)
    value = [[round(changeLetter(len(values), i + 1, j), 4) for i in range(0, 37)] for j in range(0, len(values))]
    print(value[0])
    for i in range(0, len(values)):
        for j in range(0, 37):
            if (values[i] == value[i][j]):
                result += letters[j]
    return result

line = Encrypt("The mouse flew over the nest")
x = np.linspace(0, 1, 10000);
pylab.figure()
pylab.plot(x, line)
pylab.show()
print(Decrypt(line))
